import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

class RobustGRPOPolicy(nn.Module):
    def __init__(self, obs_dim=24, action_dim=2, hidden_dim=256, lidar_rays=16):
        """
        针对不确定性环境优化的 CNN-GRU 策略网络
        
        Args:
            obs_dim (int): 环境观测空间总维度 (8 + 16 = 24)
            action_dim (int): 动作空间维度 (2: v, w)
            hidden_dim (int): GRU 和 隐藏层的宽度
            lidar_rays (int): 激光雷达的线数
        """
        super(RobustGRPOPolicy, self).__init__()
        
        self.lidar_rays = lidar_rays
        self.vector_dim = obs_dim - lidar_rays  # 24 - 16 = 8 (位置、目标等)
        
        # --- 1. 感知层 (Perception Encoder) ---
        
        # A. 激光雷达分支 (Lidar Branch) - 使用 1D CNN 提取特征
        # 相比 MLP，CNN 能更好地处理雷达的空间结构，且对单根射线的噪声不那么敏感
        self.lidar_conv = nn.Sequential(
            # Input: [Batch, 1, 16] -> Output: [Batch, 16, 16] (Padding保大)
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            # Output: [Batch, 32, 8] (Stride=2 下采样，减少计算量)
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        # 计算卷积后的扁平化维度: 32 channels * 8 width = 256
        self.lidar_out_dim = 256
        
        # B. 向量状态分支 (Vector Branch)
        self.vector_fc = nn.Sequential(
            nn.Linear(self.vector_dim, 64),
            nn.Tanh() # Tanh 通常对位置/角度归一化数据更友好
        )
        
        # C. 特征融合 (Fusion)
        self.fusion_fc = nn.Sequential(
            nn.Linear(self.lidar_out_dim + 64, hidden_dim),
            nn.ReLU()
        )

        # --- 2. 记忆层 (Memory Core) ---
        # GRU 负责处理序列信息，抵抗 GPS 噪声，预测动态障碍物轨迹
        self.gru = nn.GRU(
            input_size=hidden_dim, 
            hidden_size=hidden_dim, 
            batch_first=True
        )

        # --- 3. 决策层 (Policy Head) ---
        # 均值输出
        self.mu_head = nn.Linear(hidden_dim, action_dim)
        
        # 标准差输出 (学习 log_std 比直接学习 std 更数值稳定)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, obs, hidden_state=None):
        """
        前向传播
        
        Args:
            obs: [Batch, Seq_Len, 24] - 原始观测序列
            hidden_state: [1, Batch, Hidden] - GRU 隐状态
            
        Returns:
            mu: 动作均值
            std: 动作标准差
            next_hidden: 更新后的隐状态
        """
        # 1. 维度处理 (合并 Batch 和 Sequence 以便进行 CNN/FC 处理)
        # Obs shape: [B, T, 24] -> [B*T, 24]
        batch_size, seq_len, _ = obs.shape
        obs_flat = obs.view(-1, obs.shape[-1])
        
        # 2. 数据切分 (Slicing) 与 归一化 (Normalization)
        # 根据环境代码: 前8位是向量信息，后16位是雷达
        vector_input = obs_flat[:, :self.vector_dim]  # [B*T, 8]
        lidar_input = obs_flat[:, self.vector_dim:]   # [B*T, 16]
        
        # [关键适配] 归一化: 神经网络不喜欢 0-30 或 0-100 的大数值
        # 假设雷达最大距离 30m，地图大小 100m
        lidar_input = lidar_input / 30.0 
        # vector_input 不需要除 100，因为你的环境中 pos 已经除过 100，sin/cos 也是 -1~1
        
        # 3. 特征提取
        # Lidar 增加 channel 维: [B*T, 16] -> [B*T, 1, 16]
        lidar_feat = self.lidar_conv(lidar_input.unsqueeze(1))
        vector_feat = self.vector_fc(vector_input)
        
        # 融合
        fusion_feat = self.fusion_fc(torch.cat([vector_feat, lidar_feat], dim=1))
        
        # 4. 时序处理 (恢复序列维度)
        # [B*T, Hidden] -> [B, T, Hidden]
        gru_input = fusion_feat.view(batch_size, seq_len, -1)
        
        # GRU Forward
        gru_out, next_hidden = self.gru(gru_input, hidden_state)
        
        # 我们通常只用序列的最后一步输出来做当前决策
        # 但在训练时(GRPO计算Loss)，可能需要整个序列的输出
        # 这里为了通用性，输出整个序列的动作分布参数
        
        # 5. 动作输出
        mu = self.mu_head(gru_out)
        log_std = self.log_std_head(gru_out)
        
        # 限制 log_std 防止梯度爆炸
        log_std = torch.clamp(log_std, min=-20, max=2)
        std = torch.exp(log_std)
        
        # Tanh 激活将均值限制在 [-1, 1] (后续需在环境外缩放)
        mu = torch.tanh(mu)
        
        return mu, std, next_hidden

    def get_action(self, obs, hidden_state, deterministic=False):
        """
        推理用的辅助函数 (处理单步输入)
        obs: [Batch, 24] or [24]
        """
        # 增加序列维度 [Batch, 1, 24]
        if obs.dim() == 1:
            obs = obs.unsqueeze(0).unsqueeze(0)
        elif obs.dim() == 2:
            obs = obs.unsqueeze(1)
            
        mu, std, next_hidden = self.forward(obs, hidden_state)
        
        # 取最后一个时间步
        mu = mu[:, -1, :]
        std = std[:, -1, :]
        
        if deterministic:
            action = mu
        else:
            dist = Normal(mu, std)
            action = dist.rsample() # Reparameterization trick
            
        return action, next_hidden