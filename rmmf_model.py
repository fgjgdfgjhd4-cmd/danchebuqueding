import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """
    正交初始化权重，有助于RL训练的稳定性
    """
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class LegacyRMMF_ActorCritic(nn.Module):
    """
    兼容旧版 checkpoint 的 RMMF 结构。
    该版本使用直接拼接的 128 -> hidden_dim 融合层，
    不包含后续新增的投影、门控与 memory refine 模块。
    """
    def __init__(self, observation_dim=24, action_dim=2, hidden_dim=128):
        super(LegacyRMMF_ActorCritic, self).__init__()

        self.observation_dim = observation_dim
        self.hidden_dim = hidden_dim
        self.split_idx = 8
        self.proprio_dim = 8
        self.lidar_dim = 16

        self.proprio_net = nn.Sequential(
            layer_init(nn.Linear(self.proprio_dim, 64)),
            nn.ReLU(),
            layer_init(nn.Linear(64, 64)),
            nn.ReLU()
        )

        self.lidar_cnn = nn.Sequential(
            layer_init(nn.Conv1d(in_channels=1, out_channels=8, kernel_size=3, padding=1)),
            nn.ReLU(),
            layer_init(nn.Conv1d(in_channels=8, out_channels=16, kernel_size=3, padding=1)),
            nn.ReLU(),
            nn.Flatten()
        )
        self.cnn_out_dim = 16 * 16

        self.lidar_fc = nn.Sequential(
            layer_init(nn.Linear(self.cnn_out_dim, 64)),
            nn.ReLU()
        )

        self.fusion_net = nn.Sequential(
            layer_init(nn.Linear(64 + 64, hidden_dim)),
            nn.ReLU()
        )

        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)

        self.critic = nn.Sequential(
            layer_init(nn.Linear(hidden_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0)
        )

        self.actor_mean = layer_init(nn.Linear(hidden_dim, action_dim), std=0.01)
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim) - 0.5)

        self.register_buffer("action_scale", torch.tensor([1.5, 1.0]))
        self.register_buffer("action_bias",  torch.tensor([0.5, 0.0]))

    def _extract_features(self, x):
        proprio_input = x[:, :self.split_idx]
        lidar_input = x[:, self.split_idx:]

        proprio_feat = self.proprio_net(proprio_input)
        lidar_input = lidar_input.unsqueeze(1)
        lidar_feat_raw = self.lidar_cnn(lidar_input)
        lidar_feat = self.lidar_fc(lidar_feat_raw)

        fusion_input = torch.cat([proprio_feat, lidar_feat], dim=1)
        return self.fusion_net(fusion_input)

    def forward(self, x, hidden_state=None):
        batch_size, seq_len, _ = x.shape
        x_flat = x.reshape(-1, self.observation_dim)
        features_flat = self._extract_features(x_flat)
        features_seq = features_flat.reshape(batch_size, seq_len, self.hidden_dim)

        if hidden_state is None:
            hidden_state = torch.zeros(1, batch_size, self.hidden_dim).to(x.device)

        gru_out, next_hidden = self.gru(features_seq, hidden_state)
        return gru_out, next_hidden

    def get_action(self, x, hidden_state=None, deterministic=False):
        x = x.unsqueeze(1)
        gru_out, next_hidden = self.forward(x, hidden_state)
        belief_state = gru_out[:, -1, :]

        value = self.critic(belief_state)
        mean = torch.tanh(self.actor_mean(belief_state))
        log_std = self.actor_logstd.expand_as(mean)
        std = torch.exp(log_std)
        dist = torch.distributions.Normal(mean, std)

        if deterministic:
            raw_action = mean
        else:
            raw_action = dist.rsample()

        raw_action_clipped = torch.clamp(raw_action, -1.0, 1.0)
        log_prob = dist.log_prob(raw_action).sum(dim=-1)
        scaled_action = raw_action_clipped * self.action_scale + self.action_bias

        return scaled_action, raw_action, log_prob, next_hidden, value

    def evaluate_actions(self, obs, actions, hidden_states, masks=None):
        gru_out, _ = self.forward(obs, hidden_states)
        values = self.critic(gru_out)
        mean = torch.tanh(self.actor_mean(gru_out))
        log_std = self.actor_logstd.expand_as(mean)
        std = torch.exp(log_std)
        dist = torch.distributions.Normal(mean, std)
        log_probs = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return values, log_probs, entropy

class RMMF_ActorCritic(nn.Module):
    """
    基于循环神经网络的多模态融合架构 (RMMF)
    适配环境: UncertainComplexEnv (24维观测, 2维动作)
    """
    def __init__(self, observation_dim=24, action_dim=2, hidden_dim=128):
        super(RMMF_ActorCritic, self).__init__()
        
        # --- 1. 参数定义 ---
        self.observation_dim = observation_dim
        self.hidden_dim = hidden_dim
        # 观测空间切分点: 前8维是本体信息(位置/朝向/目标), 后16维是雷达
        self.split_idx = 8 
        self.proprio_dim = 8
        self.lidar_dim = 16

        # --- 2. 双流特征提取器 (Feature Extractors) ---
        
        # 流 A: 本体感知流 (MLP)
        # 处理位置、航向、目标距离等低维强语义信息
        self.proprio_net = nn.Sequential(
            layer_init(nn.Linear(self.proprio_dim, 64)),
            nn.ReLU(),
            layer_init(nn.Linear(64, 64)),
            nn.ReLU()
        )

        # 流 B: 空间感知流 (1D-CNN)
        # 处理激光雷达数据，提取局部几何特征
        # 输入: (Batch, 1, 16) -> 输出特征
        self.lidar_cnn = nn.Sequential(
            # 第一层卷积: 提取基础几何特征 (如墙壁、角落)
            layer_init(nn.Conv1d(in_channels=1, out_channels=8, kernel_size=3, padding=1)),
            nn.ReLU(),
            # 第二层卷积: 整合特征
            layer_init(nn.Conv1d(in_channels=8, out_channels=16, kernel_size=3, padding=1)),
            nn.ReLU(),
            nn.Flatten() # 展平: 16通道 * 16长度 = 256特征 (如果保持长度不变)
        )
        # 计算CNN输出后的维度: 16(通道) * 16(长度) = 256
        self.cnn_out_dim = 16 * 16 
        
        # 特征压缩层: 将CNN的高维特征降维，以便与本体特征融合
        self.lidar_fc = nn.Sequential(
            layer_init(nn.Linear(self.cnn_out_dim, 64)),
            nn.ReLU()
        )

        # --- 3. 多模态融合层 (Fusion Layer) ---
        # 输入: 本体特征(64) + 雷达特征(64) = 128
        self.proprio_proj = layer_init(nn.Linear(64, hidden_dim))
        self.lidar_proj = layer_init(nn.Linear(64, hidden_dim))
        self.modality_gate = nn.Sequential(
            layer_init(nn.Linear(64 + 64, hidden_dim)),
            nn.Sigmoid()
        )
        self.fusion_net = nn.Sequential(
            layer_init(nn.Linear(hidden_dim * 3, hidden_dim)),
            nn.ReLU()
        )
        self.feature_norm = nn.LayerNorm(hidden_dim)

        # --- 4. 时序信念编码器 (GRU Memory) ---
        # 核心模块: 将当前融合特征与历史记忆结合，形成 Belief State
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.memory_refine = nn.Sequential(
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh()
        )
        self.memory_gate = nn.Sequential(
            layer_init(nn.Linear(hidden_dim * 2, hidden_dim)),
            nn.Sigmoid()
        )
        self.memory_norm = nn.LayerNorm(hidden_dim)

        # --- 5. 决策头 (Decoupled Heads) ---
        
        # Critic Head: 输出状态价值 V(s)
        self.critic = nn.Sequential(
            layer_init(nn.Linear(hidden_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0)
        )

        # Actor Head: 输出动作分布参数 (Mean, LogStd)
        # Mean 使用 tanh 限制在 [-1, 1]
        self.actor_mean = layer_init(nn.Linear(hidden_dim, action_dim), std=0.01)
        # LogStd 为可学习参数，初始值设为 -0.5 (即 std ≈ 0.6)
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim) - 0.5)

        # --- 6. 动作重缩放参数 (Action Scaling) ---
        # 对应环境: v \in [-1, 2], w \in [-1, 1]
        # v: scale=(2 - (-1))/2 = 1.5, bias=(2 + (-1))/2 = 0.5
        # w: scale=(1 - (-1))/2 = 1.0, bias=(1 + (-1))/2 = 0.0
        self.register_buffer("action_scale", torch.tensor([1.5, 1.0]))
        self.register_buffer("action_bias",  torch.tensor([0.5, 0.0]))

    def _extract_features(self, x):
        """
        内部函数：执行双流特征提取与融合
        输入 x: (Batch_Size * Seq_Len, 24) -> 这里的输入已经展平了时间维度
        """
        # 1. 数据切分
        proprio_input = x[:, :self.split_idx] # [B*T, 8]
        lidar_input = x[:, self.split_idx:]   # [B*T, 16]

        # 2. 处理本体流
        proprio_feat = self.proprio_net(proprio_input)

        # 3. 处理雷达流
        # CNN 需要维度 (N, Channels, Length)
        lidar_input = lidar_input.unsqueeze(1) # [B*T, 1, 16]
        lidar_feat_raw = self.lidar_cnn(lidar_input)
        lidar_feat = self.lidar_fc(lidar_feat_raw)

        # 4. 融合
        proprio_proj = self.proprio_proj(proprio_feat)
        lidar_proj = self.lidar_proj(lidar_feat)
        gate = self.modality_gate(torch.cat([proprio_feat, lidar_feat], dim=1))
        gated_feat = gate * proprio_proj + (1.0 - gate) * lidar_proj
        fusion_input = torch.cat([proprio_proj, lidar_proj, gated_feat], dim=1)
        fused_feat = self.feature_norm(self.fusion_net(fusion_input))
        # fusion_input = torch.cat([proprio_feat, lidar_feat], dim=1)
        # fused_feat = self.fusion_net(fusion_input)
        
        return fused_feat

    def forward(self, x, hidden_state=None):
        """
        前向传播 (用于训练阶段，处理序列数据)
        x: (Batch, Seq_Len, 24)
        hidden_state: (1, Batch, Hidden_Dim)
        """
        batch_size, seq_len, _ = x.shape
        
        # 1. 展平 Batch 和 Seq 维度以进行并行特征提取
        # x_flat = x.reshape(-1, 24)
        x_flat = x.reshape(-1, self.observation_dim)
        features_flat = self._extract_features(x_flat)
        
        # 2. 恢复序列维度以输入 GRU
        features_seq = features_flat.reshape(batch_size, seq_len, self.hidden_dim)
        
        # 3. GRU 运算 (Belief State Update)
        if hidden_state is None:
            # 默认为全0状态
            hidden_state = torch.zeros(1, batch_size, self.hidden_dim).to(x.device)
            
        gru_out, next_hidden = self.gru(features_seq, hidden_state)
        memory_gate = self.memory_gate(torch.cat([features_seq, gru_out], dim=-1))
        refined_memory = self.memory_refine(features_seq)
        gru_out = self.memory_norm(gru_out + memory_gate * refined_memory)
        # gru_out: (Batch, Seq_Len, Hidden)
        
        return gru_out, next_hidden

    def get_action(self, x, hidden_state=None, deterministic=False):
        """
        推理与采样函数 (用于环境交互)
        输入 x: (Batch, 24) 当前观测 (通常 Seq_Len=1)
        返回: scaled_action, raw_action, log_prob, next_hidden, value
        """
        # 增加序列维度: (Batch, 24) -> (Batch, 1, 24)
        x = x.unsqueeze(1)
        
        # 获取 GRU 输出 (Belief State)
        gru_out, next_hidden = self.forward(x, hidden_state)
        # 取序列最后一个时间步: (Batch, Hidden)
        belief_state = gru_out[:, -1, :]
        
        # Critic 价值估计
        value = self.critic(belief_state)
        
        # Actor 动作分布
        mean = torch.tanh(self.actor_mean(belief_state)) # 原始均值在 [-1, 1]
        log_std = self.actor_logstd.expand_as(mean)
        std = torch.exp(log_std)
        
        # 构建高斯分布
        dist = torch.distributions.Normal(mean, std)
        
        if deterministic:
            raw_action = mean
        else:
            # 重参数化采样
            raw_action = dist.rsample()
            
        # [关键] 动作截断，防止采样超出 [-1, 1] 导致缩放后超出物理极限
        # 虽然 Normal 分布可能采样到 >1 的值，但我们的 scale 是基于 [-1, 1] 设计的
        # 在训练时通常保留 raw_action 计算 log_prob，但在执行时需截断
        raw_action_clipped = torch.clamp(raw_action, -1.0, 1.0)
        
        # 计算 Log Probability (用于 PPO/GRPO Loss)
        log_prob = dist.log_prob(raw_action).sum(dim=-1)
        # 注意: 这里的 log_prob 是针对 raw_action 的，不需要包含 Jacobian 修正，
        # 因为我们是在环境步进前做的线性变换，而非网络内的非线性变换。
        
        # [关键] 动作重缩放 (Affine Transformation)
        # Map [-1, 1] -> Environment Range
        scaled_action = raw_action_clipped * self.action_scale + self.action_bias
        
        return scaled_action, raw_action, log_prob, next_hidden, value

    def evaluate_actions(self, obs, actions, hidden_states, masks=None):
        """
        动作评估函数 (用于训练更新阶段)
        obs: (Batch, Seq, 24)
        actions: (Batch, Seq, 2) -> 这里的 actions 指的是 raw_actions (未缩放的)
        hidden_states: (1, Batch, Hidden)
        """
        # 获取整个序列的 Belief States
        gru_out, _ = self.forward(obs, hidden_states)
        
        # Critic Values
        values = self.critic(gru_out)
        
        # Actor Distributions
        mean = torch.tanh(self.actor_mean(gru_out))
        log_std = self.actor_logstd.expand_as(mean)
        std = torch.exp(log_std)
        dist = torch.distributions.Normal(mean, std)
        
        # 计算 Log Probs 和 Entropy
        log_probs = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        
        return values, log_probs, entropy


def detect_rmmf_variant(state_dict):
    """
    根据 checkpoint 参数名自动识别网络版本。
    """
    if "proprio_proj.weight" in state_dict:
        return "current"
    return "legacy"


def build_rmmf_model_from_state_dict(state_dict, observation_dim=24, action_dim=2, hidden_dim=128):
    """
    根据 checkpoint 自动构造匹配的 RMMF 模型。
    """
    variant = detect_rmmf_variant(state_dict)
    if variant == "legacy":
        model = LegacyRMMF_ActorCritic(
            observation_dim=observation_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
        )
    else:
        model = RMMF_ActorCritic(
            observation_dim=observation_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
        )
    return model, variant


# --- 测试代码 ---
if __name__ == "__main__":
    # 模拟环境交互测试
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RMMF_ActorCritic().to(device)
    
    print(f"Model Architecture:\n{model}")
    
    # 1. 测试推理 (Inference)
    batch_size = 1
    dummy_obs = torch.randn(batch_size, 24).to(device) # 单步观测
    hidden = None # 初始隐藏状态
    
    print("\n--- Inference Test ---")
    scaled_action, raw_action, log_prob, next_hidden, value = model.get_action(dummy_obs, hidden)
    
    print(f"Obs Shape: {dummy_obs.shape}")
    print(f"Raw Action (Net Output): {raw_action.detach().cpu().numpy()}")
    print(f"Scaled Action (Env Input): {scaled_action.detach().cpu().numpy()}")
    print(f"Action Limits Check: v={scaled_action[0,0]:.2f} (Target [-1, 2]), w={scaled_action[0,1]:.2f} (Target [-1, 1])")
    print(f"Value Estimate: {value.item():.4f}")
    print(f"Next Hidden Shape: {next_hidden.shape}")

    # 2. 测试序列训练 (Training Forward)
    print("\n--- Training Sequence Test ---")
    seq_len = 32
    batch_size = 4
    dummy_seq_obs = torch.randn(batch_size, seq_len, 24).to(device)
    dummy_seq_actions = torch.randn(batch_size, seq_len, 2).to(device) # Raw actions
    # 初始隐藏状态需匹配 Batch
    hidden_init = torch.zeros(1, batch_size, 128).to(device)
    
    values, log_probs, entropy = model.evaluate_actions(dummy_seq_obs, dummy_seq_actions, hidden_init)
    
    print(f"Input Seq Shape: {dummy_seq_obs.shape}")
    print(f"Values Shape: {values.shape} (Should be B, T, 1)")
    print(f"Log Probs Shape: {log_probs.shape} (Should be B, T)")
    print(f"Entropy Shape: {entropy.shape} (Should be B, T)")