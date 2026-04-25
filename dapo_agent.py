import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class DAPO_Algorithm:
    """
    DAPO (Decoupled Clip and Dynamic sampling Policy Optimization) 算法类
    适配路径规划环境 (UncertainComplexEnv) 与 RMMF 网络模型
    参考论文: DAPO: An Open-Source LLM Reinforcement Learning System at Scale [cite: 2262]
    """
    def __init__(
        self,
        model,
        lr=1e-6,
        eps_low=0.2,      # 下限裁剪 
        eps_high=0.28,    # 上限裁剪 (Clip-Higher 策略，鼓励探索) 
        max_steps=1000,   # L_max: 环境设定的最大步数 [cite: 2588]
        cache_len=200,    # L_cache: 用于软惩罚的缓冲长度 [cite: 2588]
        device="cuda"
    ):
        self.model = model
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr)
        self.eps_low = eps_low
        self.eps_high = eps_high
        self.max_steps = max_steps
        self.cache_len = cache_len
        self.device = device

    def soft_overlong_punishment(self, path_lengths):
        """
        实现软性超长惩罚 (Soft Overlong Punishment) [cite: 2551, 2558]
        公式参考论文 (13): 
        - 长度 <= L_max - L_cache: 0
        - L_max - L_cache < 长度 <= L_max: 线性惩罚
        - 长度 > L_max: -1
        """
        punishments = []
        l_threshold = self.max_steps - self.cache_len
        
        for length in path_lengths:
            if length <= l_threshold:
                punishments.append(0.0)
            elif length <= self.max_steps:
                # 线性增长的惩罚项 
                p = -(length - l_threshold) / self.cache_len
                punishments.append(p)
            else:
                punishments.append(-1.0)
                
        return torch.tensor(punishments, device=self.device)

    def compute_group_advantages(self, rewards):
        """
        组相对优势估计 (Group Relative Advantage Estimation) [cite: 2369, 2400]
        输入奖励形状: (Num_Prompts, Group_Size)
        """
        mean = rewards.mean(dim=1, keepdim=True)
        std = rewards.std(dim=1, keepdim=True) + 1e-8
        advantages = (rewards - mean) / std
        return advantages

    def compute_loss(self, obs, actions, old_log_probs, advantages, hidden_states, masks=None):
        """
        DAPO 核心损失函数计算 [cite: 2506]
        采用 Token-Level (步级) 权重平衡 [cite: 2313, 2505]
        以及解耦裁剪策略 (Clip-Higher) [cite: 2404, 2444]
        """
        # 1. 通过模型获取当前动作的 log_probs
        # 使用 rmmf_model 的 evaluate_actions 接口
        _, curr_log_probs, _ = self.model.evaluate_actions(obs, actions, hidden_states)
        
        # 2. 计算重要性采样比率 r_i,t(theta) [cite: 2373, 2400]
        ratio = torch.exp(curr_log_probs - old_log_probs)
        
        # 3. 实现解耦裁剪 (Decoupled Clip) [cite: 2446, 2507]
        # 当优势 A > 0 时尝试增加概率，使用 eps_high 限制上限
        # 当优势 A < 0 时尝试降低概率，使用 eps_low 限制下限
        surr1 = ratio * advantages
        
        # 定义不对称裁剪边界
        clip_ratio = torch.where(
            advantages > 0,
            torch.clamp(ratio, 1.0 - self.eps_low, 1.0 + self.eps_high), # 增加上限鼓励探索 [cite: 2450]
            torch.clamp(ratio, 1.0 - self.eps_low, 1.0 + self.eps_high)  # 下限保持稳定 [cite: 2452]
        )
        surr2 = clip_ratio * advantages
        
        # 4. Token-level Policy Gradient Loss [cite: 2506, 2507]
        # 计算所有步（tokens）的平均值，而不是序列平均
        if masks is not None:
            # 只计算有效步的损失 (处理填充)
            loss = -(torch.min(surr1, surr2) * masks).sum() / masks.sum()
        else:
            loss = -torch.min(surr1, surr2).mean()
            
        return loss

    def update(self, storage):
        """
        模型更新逻辑
        storage: 包含轨迹数据的字典，需通过 Dynamic Sampling 过滤 
        """
        # A. 提取数据
        obs = storage['obs']            # (Batch, Seq_Len, 24)
        actions = storage['actions']    # (Batch, Seq_Len, 2)
        old_log_probs = storage['log_probs']
        group_rewards = storage['rewards'] # (Num_Prompts, Group_Size)
        success_flags = storage['success'] # (Num_Prompts, Group_Size) bool 数组
        path_lengths = storage['lengths']  # 每条轨迹的长度
        h_states = storage['hidden_states']

        # B. 动态采样过滤 (Dynamic Sampling) 
        # 过滤掉组内全部成功(Accuracy=1)或全部失败(Accuracy=0)的组，因为它们的优势为0，无法提供梯度信号 
        valid_group_mask = []
        for i in range(len(success_flags)):
            num_success = success_flags[i].sum()
            group_size = len(success_flags[i])
            # 条件: 0 < 成功数 < 组大小 [cite: 2399, 2462]
            valid_group_mask.append(0 < num_success < group_size)
        
        # 只保留有效梯度的数据
        valid_indices = torch.tensor(valid_group_mask, device=self.device)
        # 此处需根据 indices 进一步筛选 obs, actions 等，具体视训练 Loop 实现而定

        # C. 奖励修正：加入软性超长惩罚 [cite: 2551, 2554]
        punishments = self.soft_overlong_punishment(path_lengths)
        adjusted_rewards = group_rewards + punishments.view_as(group_rewards)

        # D. 计算组优势 [cite: 2369, 2400]
        advantages = self.compute_group_advantages(adjusted_rewards)
        # 将优势广播到每个时间步 [cite: 2400]
        flat_advantages = advantages.view(-1, 1).expand(-1, obs.size(1))

        # E. 计算损失并更新
        self.optimizer.zero_grad()
        loss = self.compute_loss(obs, actions, old_log_probs, flat_advantages, h_states)
        loss.backward()
        
        # 防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
        self.optimizer.step()

        return loss.item()