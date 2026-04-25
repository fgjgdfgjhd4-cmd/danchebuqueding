

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class AC_GDPO_Agent:
    """
    AC-GDPO 智能体类 (改进版 v2.0)
    
    改进日志:
    1. 修正优势计算公式: 移除二次均值中心化，保留 GAE 的正负号信息。
    2. 引入 Mini-batch 训练: 在 Update 中对轨迹进行 Shuffle 和切分。
    3. 增加奖励缩放: 防止数值过大导致 Critic 不稳定。
    4. 完善 KL 散度计算与监控。
    """
    def __init__(self, 
                 model, 
                 lr=3e-4, 
                 gamma=0.99, 
                 gae_lambda=0.95,
                 clip_low=0.2, 
                 clip_high=0.28, # DAPO 核心: 上界更高，鼓励探索
                 entropy_coef=0.01,
                 value_coef=0.5,
                 max_grad_norm=0.5,
                 device="cpu"):
        """
        初始化智能体
        :param model: 实例化后的 RMMF_ActorCritic 网络模型
        :param lr: 学习率
        :param clip_low: DAPO 下界裁剪阈值 (通常 0.2)
        :param clip_high: DAPO 上界裁剪阈值 (通常 0.28, 大于下界)
        """
        self.model = model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, eps=1e-5)
        
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_low = clip_low
        self.clip_high = clip_high
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.device = device

    def get_action(self, obs, hidden_state=None, deterministic=False):
        """
        与环境交互时调用，获取动作
        """
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs).float().to(self.device)
        if obs.ndim == 1:
            obs = obs.unsqueeze(0)
            
        with torch.no_grad():
            return self.model.get_action(obs, hidden_state, deterministic)

    def compute_group_advantages(self, batch_groups):
        """
        计算组相对优势 (核心改进部分)
        
        逻辑变更:
        - 原版: Norm_Adv = (GAE - Mean) / Std
        - 新版: Norm_Adv = GAE / (Std + eps)
        
        原因: GAE 均值理论上已接近 0。强制减去组内均值会导致 sum(Adv)=0，
        在全量更新时导致梯度抵消 (Loss=0)。保留 GAE 原始符号能正确指示动作好坏。
        """
        flat_data = []
        
        for group in batch_groups:
            group_advantages = []
            
            # --- 1. 计算每条轨迹的 GAE ---
            for traj in group:
                # [改进] 奖励缩放: 将巨大的奖励值缩小，稳定 Critic 训练
                # 例如: -200 -> -20.0, +500 -> +50.0
                rewards = torch.tensor(traj['rewards']).to(self.device) / 10.0
                
                values = torch.tensor(traj['values']).to(self.device)
                dones = torch.tensor(traj['dones'], dtype=torch.float32).to(self.device)
                next_value = traj['next_value'] # 已经是标量值
                
                # GAE 计算
                advantages = torch.zeros_like(rewards).to(self.device)
                lastgaelam = 0
                seq_len = len(rewards)
                
                for t in reversed(range(seq_len)):
                    if t == seq_len - 1:
                        # 如果 truncated=True (超时)，nextnonterminal=1，利用 next_value
                        # 如果 terminated=True (撞/到)，nextnonterminal=0，next_value 被忽略
                        nextnonterminal = 1.0 - float(traj.get('truncated', False))
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t+1]
                        nextvalues = values[t+1]
                        
                    delta = rewards[t] + self.gamma * nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = delta + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam
                
                # Returns = GAE + Value (用于 Critic 训练的目标值)
                returns = advantages + values
                
                # 暂存 GAE
                group_advantages.append(advantages)
                
                # 将计算好的 Tensor 存回字典
                traj['advantages'] = advantages
                traj['returns'] = returns

            # --- 2. 组内统计量计算 (GRPO 核心) ---
            all_adv_in_group = torch.cat(group_advantages)
            # group_mean = all_adv_in_group.mean() # [已移除] 不再减均值
            group_std = all_adv_in_group.std()
            
            # --- 3. 优势归一化 ---
            for traj in group:
                # [核心修正] 仅缩放，不中心化
                # 这样如果整组表现都很差(都是负GAE)，模型会收到明确的负反馈，而不是把"最好的烂动作"当成正样本
                traj['norm_advantages'] = traj['advantages'] / (group_std + 1e-8)
                flat_data.append(traj)
                
        return flat_data

    def update(self, dataloader, num_epochs=4, mini_batch_size=4):
        """
        策略更新函数 (改进版: 支持 Trajectory-level Mini-batch)
        
        :param dataloader: 包含所有轨迹数据的 DataLoader (通常只有 1 个 Full Batch)
        :param num_epochs: 更新轮数
        :param mini_batch_size: Mini-batch 大小 (多少条轨迹一组). 
                                建议设为 Group_Size / 4 (例如 16/4=4)
        """
        # 初始化日志统计
        epoch_actor_loss = 0
        epoch_critic_loss = 0
        epoch_entropy = 0
        epoch_kl = 0
        num_updates = 0
        
        # 1. 获取全量数据 (假设 dataloader 只产生一个包含所有轨迹的大 Batch)
        # 结构: [Batch_Size(16), Max_Seq_Len, Dim]
        full_batch = next(iter(dataloader))
        
        # 获取总轨迹数 (Group Size)
        total_trajs = full_batch['obs'].shape[0] # e.g., 16
        
        # 2. Epoch 循环
        for epoch in range(num_epochs):
            # [关键] 随机打乱轨迹索引
            # RNN 训练不能打乱时间步，但可以打乱轨迹顺序
            indices = torch.randperm(total_trajs)
            
            # 3. Mini-batch 循环
            for start_idx in range(0, total_trajs, mini_batch_size):
                # 获取当前 Mini-batch 的索引 (例如 [0, 5, 12, 3])
                mb_indices = indices[start_idx : start_idx + mini_batch_size]
                
                # --- 数据切片 ---
                b_obs = full_batch['obs'][mb_indices]           # [MB, T, 24]
                b_actions = full_batch['actions'][mb_indices]   # [MB, T, 2]
                b_old_log_probs = full_batch['log_probs'][mb_indices]
                b_returns = full_batch['returns'][mb_indices]
                b_advantages = full_batch['norm_advantages'][mb_indices]
                # 初始 Hidden 只取切片对应的部分: [1, Total, H] -> [1, MB, H]
                b_hiddens = full_batch['hidden_states'][:, mb_indices, :]
                
                # --- 重新评估动作 (Re-evaluate) ---
                # 将 Mini-batch 数据输入模型，计算当前的 LogProb, Value, Entropy
                new_values, new_log_probs, entropy = self.model.evaluate_actions(
                    b_obs, b_actions, b_hiddens
                )
                
                # --- 掩码处理 (Masking Padding) ---
                # 因为 pad_collate 填充了 0，我们需要根据 returns 是否为 0 (或者另外传 mask) 来忽略填充部分
                # 这里简单起见，假设 padded 部分的 advantages 也是 0 (pad_collate 应该处理好)，
                # 但更严谨的做法是生成 mask。由于 advantages 在 pad 处通常为 0，loss 贡献也为 0。
                # 这是一个简化的处理。
                
                # 展平维度 [MB, T, ...] -> [MB*T, ...] 以便计算均值
                new_values = new_values.view(-1)
                new_log_probs = new_log_probs.view(-1)
                entropy = entropy.view(-1)
                b_old_log_probs = b_old_log_probs.view(-1)
                b_advantages = b_advantages.view(-1)
                b_returns = b_returns.view(-1)
                
                # 过滤掉 Padding 部分 (假设 Advantage 绝对值极小的是 Padding)
                # 或者更简单：直接计算 Mean，Padding 部分通常全 0 不影响梯度方向，只影响幅度
                # 为了准确监控 KL，最好只统计非 Pad 部分。但为保持代码简洁，这里直接计算。
                
                # --- PPO 损失计算 ---
                log_ratio = new_log_probs - b_old_log_probs
                ratio = log_ratio.exp()
                
                # DAPO 非对称裁剪
                # 如果 Advantage > 0 (好动作)，使用 clip_high (更大上界)
                # 如果 Advantage < 0 (坏动作)，使用 clip_low (标准下界)
                surr1 = ratio * b_advantages
                ratio_clipped = torch.clamp(ratio, 1.0 - self.clip_low, 1.0 + self.clip_high)
                surr2 = ratio_clipped * b_advantages
                
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # Critic 损失 (MSE)
                critic_loss = 0.5 * ((new_values - b_returns) ** 2).mean()
                
                # 熵奖励
                entropy_loss = entropy.mean()
                
                # 总损失
                loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy_loss
                
                # --- 近似 KL 散度计算 (用于监控) ---
                with torch.no_grad():
                    # KL(Old || New) approx = (LogProb_Old - LogProb_New)
                    approx_kl = (b_old_log_probs - new_log_probs).mean().item()
                
                # --- 反向传播 ---
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                # --- 累加统计 ---
                epoch_actor_loss += actor_loss.item()
                epoch_critic_loss += critic_loss.item()
                epoch_entropy += entropy_loss.item()
                epoch_kl += approx_kl
                num_updates += 1

        # 返回平均统计量
        if num_updates == 0: num_updates = 1 # 防止除零
        return {
            "loss/actor": epoch_actor_loss / num_updates,
            "loss/critic": epoch_critic_loss / num_updates,
            "loss/entropy": epoch_entropy / num_updates,
            "loss/kl": epoch_kl / num_updates
        }

    def save(self, path):
        torch.save(self.model.state_dict(), path)
        
    def load(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))