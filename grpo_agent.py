import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Normal

class GRPO_Algorithm:
    """
    针对路径规划环境 (UncertainComplexEnv) 适配的 GRPO 算法类。
    参考论文: DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models
    
    适配要点:
    1. 去中心化: 不再依赖 rmmf_model 中的 critic 网络，仅使用 Actor 部分 [cite: 56]。
    2. 组相对优势: 对同一组内的多个轨迹进行奖励归一化 [cite: 349, 374]。
    3. 显式 KL 约束: 使用论文推荐的无偏 KL 散度估计算法 。
    """
    def __init__(
        self,
        model,
        ref_model,
        lr=1e-6,
        beta=0.04,        # KL 惩罚系数
        epsilon=0.2,     # PPO 裁剪范围
        group_size=16,   # 每组采样的数量 (对应论文中的 G)
        device="cuda"
    ):
        self.model = model
        self.ref_model = ref_model # 参考模型 (通常是 SFT 后的模型快照)，需设为 eval 模式
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        self.beta = beta
        self.epsilon = epsilon
        self.group_size = group_size
        self.device = device

        # 确保参考模型不更新梯度
        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False

    def compute_group_advantages(self, rewards):
        """
        实现 GRPO 的核心：组内相对奖励归一化 [cite: 374]。
        输入 rewards 形状: (Batch_Size, Group_Size) -> 对应同一状态下采集的 G 个结果
        """
        # 计算组内均值和标准差
        mean_r = rewards.mean(dim=1, keepdim=True)
        std_r = rewards.std(dim=1, keepdim=True) + 1e-8
        
        # 归一化奖励作为优势 A_hat
        advantages = (rewards - mean_r) / std_r
        return advantages

    def update(self, obs, actions, rewards, hidden_states):
        """
        GRPO 参数更新逻辑
        :param obs: 观测序列 (Batch, Seq_Len, 24)
        :param actions: 执行的原始动作 (Batch, Seq_Len, 2)
        :param rewards: 轨迹的总奖励或每步奖励 (Batch,) 
        :param hidden_states: GRU 初始隐藏状态 (1, Batch, Hidden_Dim)
        """
        # 1. 计算当前策略的分布
        # 获取 GRU 输出的特征序列 (Batch, Seq_Len, Hidden)
        gru_out, _ = self.model(obs, hidden_states)
        mean = torch.tanh(self.model.actor_mean(gru_out))
        std = torch.exp(self.model.actor_logstd.expand_as(mean))
        curr_dist = Normal(mean, std)
        curr_log_probs = curr_dist.log_prob(actions).sum(dim=-1)

        # 2. 计算旧策略(old_policy)或参考策略(ref_policy)的分布 [cite: 293]
        with torch.no_grad():
            ref_gru_out, _ = self.ref_model(obs, hidden_states)
            ref_mean = torch.tanh(self.ref_model.actor_mean(ref_gru_out))
            ref_std = torch.exp(self.ref_model.actor_logstd.expand_as(ref_mean))
            ref_dist = Normal(ref_mean, ref_std)
            ref_log_probs = ref_dist.log_prob(actions).sum(dim=-1)

        # 3. 计算组优势 (Group Advantages) [cite: 361]
        # 假设输入时已经将相同起始状态的轨迹排列在一起，形状重塑为 (B', G)
        # B' 为独立状态数，G 为组大小
        B_total = rewards.shape[0]
        num_groups = B_total // self.group_size
        
        # 将奖励重塑为组格式以进行归一化
        group_rewards = rewards.view(num_groups, self.group_size)
        group_advantages = self.compute_group_advantages(group_rewards)
        # 展平回 (Batch,) 以便与 log_probs 相乘
        flat_advantages = group_advantages.view(-1).unsqueeze(-1) # (Batch, 1)

        # 4. 计算 PPO 裁剪损失 (Surrogate Objective) [cite: 358]
        ratio = torch.exp(curr_log_probs - ref_log_probs.detach())
        # 注意：此处优势应用在整个序列或每个 Token 上，路径规划通常按步应用
        surr1 = ratio * flat_advantages
        surr2 = torch.clamp(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon) * flat_advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # 5. 计算无偏 KL 散度约束 
        # 公式: exp(log_ref - log_curr) - (log_ref - log_curr) - 1
        log_ratio = ref_log_probs - curr_log_probs
        kl_div = torch.exp(log_ratio) - log_ratio - 1
        kl_loss = kl_div.mean()

        # 6. 总损失 = 策略损失 + beta * KL 损失 [cite: 359]
        # GRPO 不包含传统的 Value Loss (Critic Loss)
        total_loss = policy_loss + self.beta * kl_loss

        # 更新网络
        self.optimizer.zero_grad()
        total_loss.backward()
        # 建议增加梯度裁剪以适应复杂环境
        nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        return {
            "total_loss": total_loss.item(),
            "policy_loss": policy_loss.item(),
            "kl_loss": kl_loss.item(),
            "avg_advantage": group_advantages.mean().item()
        }

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def sync_ref_model(self):
        """
        定期将当前策略同步到参考模型 (用于迭代强化学习) [cite: 386]
        """
        self.ref_model.load_state_dict(self.model.state_dict())
        self.ref_model.eval()