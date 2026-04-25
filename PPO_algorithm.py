import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
from torch.nn.utils.rnn import pad_sequence

# --- 导入您现有的网络定义 ---
# 假设您的网络文件名为 CNN_GRU_policy.py
from CNN_GRU_policy import RobustGRPOPolicy

class RobustCritic(nn.Module):
    """
    PPO 需要一个 Critic (价值网络) 来估计状态价值 V(s)。
    为了公平对比，Critic 的结构设计得与 Actor (RobustGRPOPolicy) 几乎一致。
    """
    def __init__(self, obs_dim=24, hidden_dim=256, lidar_rays=16):
        super(RobustCritic, self).__init__()
        
        self.lidar_rays = lidar_rays
        self.vector_dim = obs_dim - lidar_rays
        
        # --- 1. 感知层 (与 Actor 保持一致) ---
        self.lidar_conv = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        self.lidar_out_dim = 256
        
        self.vector_fc = nn.Sequential(
            nn.Linear(self.vector_dim, 64),
            nn.Tanh()
        )
        
        self.fusion_fc = nn.Sequential(
            nn.Linear(self.lidar_out_dim + 64, hidden_dim),
            nn.ReLU()
        )

        # --- 2. 记忆层 (GRU) ---
        self.gru = nn.GRU(
            input_size=hidden_dim, 
            hidden_size=hidden_dim, 
            batch_first=True
        )

        # --- 3. 价值头 (Value Head) ---
        # 输出一个标量 V(s)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, obs, hidden_state=None):
        """
        前向传播: 计算状态价值 V(s)
        """
        # 1. 维度处理
        batch_size, seq_len, _ = obs.shape
        obs_flat = obs.view(-1, obs.shape[-1])
        
        # 2. 数据切分与归一化 (保持与 Actor 一致)
        vector_input = obs_flat[:, :self.vector_dim]
        lidar_input = obs_flat[:, self.vector_dim:] / 30.0 
        
        # 3. 特征提取
        lidar_feat = self.lidar_conv(lidar_input.unsqueeze(1))
        vector_feat = self.vector_fc(vector_input)
        fusion_feat = self.fusion_fc(torch.cat([vector_feat, lidar_feat], dim=1))
        
        # 4. 时序处理
        gru_input = fusion_feat.view(batch_size, seq_len, -1)
        gru_out, next_hidden = self.gru(gru_input, hidden_state)
        
        # 5. 价值输出 [Batch, Seq, 1]
        values = self.value_head(gru_out)
        
        return values, next_hidden

class PPO:
    """
    PPO 算法类 (Proximal Policy Optimization)
    支持 RNN (GRU) 和 变长序列更新
    """
    def __init__(self, 
                 actor_net: RobustGRPOPolicy,
                 lr_actor=3e-4,
                 lr_critic=1e-3,
                 gamma=0.99,
                 gae_lambda=0.95,
                 clip_ratio=0.2,
                 target_kl=0.01,
                 entropy_coef=0.01,
                 value_coef=0.5,
                 train_epochs=10,
                 device='cuda'):
        
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.target_kl = target_kl
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.train_epochs = train_epochs

        # --- 初始化网络 ---
        self.actor = actor_net.to(device)
        # 初始化 Critic (参数与 Actor 配置一致)
        self.critic = RobustCritic(obs_dim=24, hidden_dim=256).to(device)

        # --- 优化器 ---
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        # 损失函数
        self.mse_loss = nn.MSELoss()

    def select_action(self, obs, hidden_actor, hidden_critic, deterministic=False):
        """
        推理阶段：选择动作并评估价值
        """
        with torch.no_grad():
            # 确保输入维度正确 [1, 1, 24]
            if obs.dim() == 2:
                obs = obs.unsqueeze(1)
            
            obs = obs.to(self.device)
            if hidden_actor is not None: hidden_actor = hidden_actor.to(self.device)
            if hidden_critic is not None: hidden_critic = hidden_critic.to(self.device)

            # 1. Actor 前向
            mu, std, next_hidden_actor = self.actor(obs, hidden_actor)
            mu = mu[:, -1, :]
            std = std[:, -1, :]
            
            dist = Normal(mu, std)
            if deterministic:
                action = mu
            else:
                action = dist.sample()
            
            log_prob = dist.log_prob(action).sum(dim=-1)
            
            # 2. Critic 前向 (计算 V(s))
            value, next_hidden_critic = self.critic(obs, hidden_critic)
            value = value[:, -1, 0] # 取最后一步的标量值

            return (action.cpu().numpy(), 
                    log_prob.cpu().numpy(), 
                    value.cpu().numpy(), 
                    next_hidden_actor, 
                    next_hidden_critic)

    def update(self, rollouts):
        """
        PPO 更新核心逻辑
        Args:
            rollouts: List[Dict], 包含变长的轨迹数据
        """
        # --- 1. 数据预处理 (Padding & Tensor conversion) ---
        # 由于是 RNN，我们通常需要处理整个序列。为了批处理，我们将变长序列 Pad 到最大长度。
        
        lengths = [len(traj['reward']) for traj in rollouts]
        max_len = max(lengths)
        batch_size = len(rollouts)
        
        # 初始化 Padded Tensors
        # Obs: [Batch, MaxLen, 24]
        padded_obs = torch.zeros(batch_size, max_len, 24).to(self.device)
        padded_actions = torch.zeros(batch_size, max_len, 2).to(self.device)
        padded_old_log_probs = torch.zeros(batch_size, max_len).to(self.device)
        padded_rewards = torch.zeros(batch_size, max_len).to(self.device)
        padded_dones = torch.zeros(batch_size, max_len).to(self.device) # 用于 GAE
        mask = torch.zeros(batch_size, max_len).to(self.device) # 标记有效数据
        
        # 收集初始隐状态
        h_actor_init = torch.cat([traj['hidden_actor'] for traj in rollouts], dim=1).to(self.device)
        h_critic_init = torch.cat([traj['hidden_critic'] for traj in rollouts], dim=1).to(self.device)

        # 填充数据
        for i, traj in enumerate(rollouts):
            L = lengths[i]
            padded_obs[i, :L, :] = traj['obs']
            padded_actions[i, :L, :] = traj['action']
            padded_rewards[i, :L] = traj['reward']
            padded_old_log_probs[i, :L] = traj['log_prob']
            # done 标记: 除了最后一步可能是 True，中间都是 False
            # 注意: 如果是超时(truncate)，通常视作 done=False 以便 bootstrap value，
            # 但这里简化处理，假设最后一步 done=True 代表终止
            padded_dones[i, L-1] = 1.0 
            mask[i, :L] = 1.0

        # --- 2. 计算 GAE (Generalized Advantage Estimation) ---
        # 我们需要通过 Critic 计算整个序列的 V(s)
        with torch.no_grad():
            values, _ = self.critic(padded_obs, h_critic_init)
            values = values.squeeze(-1) * mask # [Batch, MaxLen]
        
        advantages = torch.zeros_like(padded_rewards).to(self.device)
        returns = torch.zeros_like(padded_rewards).to(self.device)
        
        # 倒序计算 GAE
        # 注意：对于 RNN Pad 过的部分，我们不计算
        last_gae_lam = 0
        for t in reversed(range(max_len)):
            if t == max_len - 1:
                next_non_terminal = 1.0 - padded_dones[:, t]
                next_value = 0 # 简化：假设 episode 结束 value 为 0 (或者可以在 rollout 里存 next_value)
            else:
                next_non_terminal = 1.0 - padded_dones[:, t] # Mask[t+1] 已经涵盖了有效性
                next_value = values[:, t+1]
            
            # delta = r + gamma * V(s') - V(s)
            delta = padded_rewards[:, t] + self.gamma * next_value * next_non_terminal - values[:, t]
            
            # gae = delta + gamma * lambda * gae_next
            # 乘以 mask[:, t] 确保填充部分的 gae 为 0
            advantages[:, t] = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            advantages[:, t] *= mask[:, t]
            last_gae_lam = advantages[:, t]
        
        # TD(lambda) Returns = Advantage + Value
        returns = advantages + values

        # 优势归一化 (减小方差)
        # 只对 mask=1 的部分进行归一化
        active_adv = advantages[mask.bool()]
        adv_mean = active_adv.mean()
        adv_std = active_adv.std() + 1e-8
        advantages = (advantages - adv_mean) / adv_std * mask # 保持 mask

        # --- 3. PPO 更新循环 ---
        for i_epoch in range(self.train_epochs):
            # 重新评估当前策略和价值
            # 注意：在 RNN 中，必须每次从头传入初始 hidden state 跑一遍序列
            # 这样才能得到这一轮更新后的 log_prob 和 value
            
            new_mu, new_std, _ = self.actor(padded_obs, h_actor_init)
            new_values, _ = self.critic(padded_obs, h_critic_init)
            new_values = new_values.squeeze(-1) * mask
            
            dist = Normal(new_mu, new_std)
            new_log_probs = dist.log_prob(padded_actions).sum(dim=-1) * mask
            entropy = dist.entropy().sum(dim=-1) * mask
            
            # 计算 Ratio
            # ratio = exp(new - old)
            ratio = torch.exp(new_log_probs - padded_old_log_probs) * mask
            
            # --- 计算 Actor Loss (Clip) ---
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * advantages
            actor_loss = -torch.min(surr1, surr2).sum() / mask.sum()
            
            # --- 计算 Critic Loss (MSE) ---
            # 也可以加 Clip，但通常直接 MSE 足够
            value_loss = 0.5 * ((new_values - returns)**2 * mask).sum() / mask.sum()
            
            # --- 总 Loss ---
            loss = actor_loss + self.value_coef * value_loss - self.entropy_coef * (entropy.sum() / mask.sum())
            
            # --- 反向传播 ---
            self.optimizer_actor.zero_grad()
            self.optimizer_critic.zero_grad()
            loss.backward()
            
            # 梯度裁剪 (防止 RNN 梯度爆炸)
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
            
            self.optimizer_actor.step()
            self.optimizer_critic.step()
            
            # (可选) 提前停止：如果 KL 散度太大，停止本轮更新
            with torch.no_grad():
                approx_kl = (padded_old_log_probs - new_log_probs).mean().item()
            if approx_kl > 1.5 * self.target_kl:
                # print(f"Early stopping at epoch {i_epoch} due to KL {approx_kl:.4f}")
                break

        return {
            "loss": loss.item(),
            "actor_loss": actor_loss.item(),
            "critic_loss": value_loss.item(),
            "entropy": (entropy.sum() / mask.sum()).item(),
            "kl": approx_kl
        }

    def save(self, path):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict()
        }, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])