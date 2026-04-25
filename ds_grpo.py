# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F
# from torch.distributions import Normal
# import numpy as np

# class DSGRPO:
#     """
#     DS-GRPO 算法核心类
#     (Dynamic Safety-aware Group Relative Policy Optimization)
    
#     包含:
#     1. 策略网络管理 (Actor)
#     2. 动态采样逻辑 (DAPO Idea)
#     3. 组内安全过滤 (Safety Filtering)
#     4. GRPO 损失函数计算与更新
#     """
#     def __init__(self, 
#                  policy_net, 
#                  optimizer_lr=3e-4, 
#                  beta_kl=0.01, 
#                  clip_ratio=0.2,
#                  device='cuda'):
#         """
#         初始化算法
        
#         Args:
#             policy_net (nn.Module): 实例化的 RobustGRPOPolicy 网络
#             optimizer_lr (float): 学习率
#             beta_kl (float): KL 散度惩罚系数 (GRPO 核心参数)
#             clip_ratio (float): PPO 风格的截断范围 (可选增强稳定性)
#             device (str): 计算设备
#         """
#         self.policy = policy_net.to(device)
#         self.optimizer = optim.Adam(self.policy.parameters(), lr=optimizer_lr)
        
#         self.beta = beta_kl
#         self.clip_ratio = clip_ratio
#         self.device = device
        
#         # 物理常数 (用于安全过滤)
#         self.dt = 0.5         # 环境时间步长
#         self.agent_radius = 1.0 # 智能体半径估计 (含安全余量)

#     def select_action(self, obs, hidden_state, deterministic=False):
#         """
#         选择动作 (推理阶段)
        
#         Returns:
#             action: 实际执行的动作 (未缩放)
#             action_log_prob: 动作的对数概率
#             next_hidden: 更新后的 GRU 隐状态
#             entropy: 策略熵 (用于不确定性评估)
#         """
#         with torch.no_grad():
#             # 确保输入维度正确 [1, 1, 24]
#             if obs.dim() == 1:
#                 obs = obs.unsqueeze(0).unsqueeze(0)
#             elif obs.dim() == 2:
#                 obs = obs.unsqueeze(1)
            
#             obs = obs.to(self.device)
#             if hidden_state is not None:
#                 hidden_state = hidden_state.to(self.device)

#             # 网络前向传播
#             mu, std, next_hidden = self.policy(obs, hidden_state)
            
#             # 只有最后一帧用于决策
#             mu = mu[:, -1, :]
#             std = std[:, -1, :]
            
#             dist = Normal(mu, std)
            
#             if deterministic:
#                 action = mu
#             else:
#                 action = dist.sample()
                
#             action_log_prob = dist.log_prob(action).sum(dim=-1)
#             entropy = dist.entropy().sum(dim=-1)
            
#             # entropy 的形状是 [16]，我们需要它的平均值作为 DAPO 的指标
#             mean_entropy = entropy.mean().item()
            
#             # 必须返回所有环境的动作和概率，不要加 [0] !
#             return action.cpu().numpy(), action_log_prob.cpu().numpy(), next_hidden, mean_entropy

#     def determine_group_size(self, uncertainty_metric, g_min=4, g_max=32):
#         """
#         [DAPO 核心] 根据不确定性动态决定采样组大小
        
#         Args:
#             uncertainty_metric (float): 通常是策略熵或雷达密集度
#             g_min (int): 最小采样数 (简单环境)
#             g_max (int): 最大采样数 (复杂环境)
        
#         Returns:
#             int: 推荐的 Group Size
#         """
#         # 简单的线性映射逻辑
#         # 假设熵的范围大约在 0.0 ~ 3.0 之间
#         normalized_u = np.clip(uncertainty_metric / 3.0, 0.0, 1.0)
#         group_size = int(g_min + (g_max - g_min) * normalized_u)
#         return group_size

#     def safety_filter(self, obs_batch, action_batch, rewards_batch):
#         """
#         [Safety Filter 核心] 基于物理模型的组内安全过滤 (修复版)
        
#         功能:
#         对预测会发生碰撞的动作进行 Advantage 惩罚。
#         修复了之前的维度广播错误，现在支持处理整个时间序列的数据。
        
#         Args:
#             obs_batch: [Batch, Seq, 24] 
#                 - 观测数据序列 (包含 Batch 和 Sequence 维度)
#             action_batch: [Batch, Seq, 2] 
#                 - 动作序列 (网络输出的原始动作 [-1, 1])
#             rewards_batch: [Batch, Seq] 
#                 - 奖励序列 (必须是 Tensor 格式以便原地修改)
            
#         Returns:
#             adjusted_rewards: [Batch, Seq] 修正后的奖励序列
#             mask: [Batch, Seq] 哪些时间步被判定为危险 (bool)
#         """
#         # --- 1. 数据提取 (处理序列维度) ---
        
#         # 提取雷达数据
#         # obs_batch 形状: [Batch, Seq, 24]
#         # 我们取最后16维作为雷达数据 -> [Batch, Seq, 16]
#         # 注意: 假设外部训练循环已确保传入的是真实物理距离(0-30m)，而非归一化数值
#         lidar_data = obs_batch[:, :, 8:] 
        
#         # 动作反归一化
#         # action_batch 形状: [Batch, Seq, 2]
#         # 取第0维(线速度 v), 范围 [-1, 1] -> [Batch, Seq]
#         v_scaled = action_batch[:, :, 0]
        
#         # 映射回真实物理速度: v: [-1, 1] -> [-1, 2] -> [-1, 2] (根据环境设定)
#         # 公式: v_real = (v_norm + 1) / 2 * (max - min) + min
#         # 这里假设环境 v 范围是 [-1, 2] (跨度3.0, 偏移-1.0)
#         v_real = (v_scaled + 1.0) / 2.0 * 3.0 - 1.0
        
#         # --- 2. 物理前瞻预测 (Physics Lookahead) ---
        
#         # 获取每个时间步、每个环境中最危险(最近)的障碍物距离
#         # lidar_data: [Batch, Seq, 16] -> 在 dim=2 (雷达线束维度) 上取最小
#         # min_lidar: [Batch, Seq]
#         min_lidar, _ = torch.min(lidar_data, dim=2) 
        
#         # 预测在当前速度下的行驶距离 (例如 0.5s 内)
#         # pred_dist: [Batch, Seq]
#         pred_dist = v_real * self.dt
        
#         # --- 3. 危险判定条件 (Element-wise Operation) ---
        
#         # 此时所有变量维度均为 [Batch, Seq]，可以直接进行逻辑运算，不会报错
#         # 条件A: 预测行驶距离 > (最近障碍物距离 - 安全半径) -> 说明刹不住或即将碰撞
#         # 条件B: 真实速度 > 0.1 -> 说明是向前开 (倒车或静止通常不视为主动撞击风险)
#         is_dangerous = (pred_dist > (min_lidar - self.agent_radius)) & (v_real > 0.1)
        
#         # --- 4. 奖励修正 (Advantage Penalty) ---
        
#         # 克隆原始奖励，避免破坏原始数据记录
#         adjusted_rewards = rewards_batch.clone()
        
#         # 定义惩罚力度 (Hard Penalty)
#         penalty = -50.0 
        
#         # 利用掩码(mask)进行原地修改
#         # 对被判定为 dangerous 的时间步，直接扣除 50 分
#         adjusted_rewards[is_dangerous] += penalty
        
#         return adjusted_rewards, is_dangerous

#     # def update(self, rollouts):
#     #     """
#     #     GRPO 核心更新逻辑
        
#     #     Args:
#     #         rollouts (list): 包含一组轨迹数据的字典列表
#     #         每个元素包含: 'obs', 'action', 'reward', 'log_prob', 'hidden'
#     #     """
#     #     # 1. 数据整理 (Collate Data)
#     #     # 假设 rollouts 是一个 Batch 的数据
#     #     obs_batch = torch.stack([x['obs'] for x in rollouts]).to(self.device) # [B, Seq, 24]
#     #     action_batch = torch.stack([x['action'] for x in rollouts]).to(self.device) # [B, 2]
#     #     reward_batch = torch.stack([torch.tensor(x['reward']) for x in rollouts]).to(self.device) # [B]
#     #     old_log_probs = torch.stack([torch.tensor(x['log_prob']) for x in rollouts]).to(self.device) # [B]
        
#     #     # 处理 GRU 隐状态 (取每条轨迹开始时的隐状态)
#     #     # 注意: 这里假设 rollouts 是并行的 Episode 片段
#     #     hiddens = torch.cat([x['hidden'] for x in rollouts], dim=1).to(self.device) # [1, B, Hidden]

#     #     # 2. 组内安全过滤 (Group-wise Safety Filtering)
#     #     # 在计算优势之前，先根据物理规则修正奖励
#     #     # 注意: 需要确保传入的 obs 包含未归一化的雷达数据用于判断，或者在这里做反归一化
#     #     # 假设 obs_batch 中的雷达数据已经是归一化过的 (0-1)，需还原为 0-30m
#     #     obs_for_safety = obs_batch.clone()
#     #     obs_for_safety[:, :, 8:] *= 30.0 
        
#     #     adjusted_rewards, mask = self.safety_filter(obs_for_safety, action_batch, reward_batch)

#     #     # 3. 组内相对优势计算 (Group Relative Advantage)
#     #     # GRPO 的精髓: A_i = (R_i - Mean(Group)) / Std(Group)
#     #     # 这里我们将整个 Batch 视为一个 Group (或者多个 Group 的集合)
#     #     # 如果 Batch 来自不同的 Start Pos，应该按 Group 分组归一化。
#     #     # 为简化实现，假设 Batch 内的数据是同分布采样的 (DAPO 思想)
        
#     #     mean_r = adjusted_rewards.mean()
#     #     std_r = adjusted_rewards.std() + 1e-8
#     #     advantages = (adjusted_rewards - mean_r) / std_r
        
#     #     # 4. 策略网络前向传播 (Re-evaluate)
#     #     # 获取当前的动作分布，用于计算 Ratio 和 KL
#     #     mu, std, _ = self.policy(obs_batch, hiddens)
#     #     # 取最后一步
#     #     # mu = mu[:, -1, :]
#     #     # std = std[:, -1, :]
#     #     dist = Normal(mu, std)
        
#     #     # 计算新的 Log Probability
#     #     new_log_probs = dist.log_prob(action_batch).sum(dim=-1)
        
#     #     # 计算 KL 散度 (近似计算: log(p_old) - log(p_new)) 
#     #     # 或者准确计算 D_KL(pi_new || pi_ref)
#     #     # 这里使用 DeepSeek-R1 风格的近似:
#     #     # approx_kl = (old_log_probs - new_log_probs) # 简化版
#     #     # 更加标准的 KL(New || Old) 计算:
#     #     # 这里的 pi_ref 我们简单地使用 old_policy (即 rollout 时的策略)
#     #     # KL(N(mu, std) || N(mu_old, std_old))
#     #     # 为简化计算，直接使用 log_ratio 近似
#     #     ratio = torch.exp(new_log_probs - old_log_probs)
#     #     approx_kl = old_log_probs - new_log_probs # log(p_old / p_new)

#     #     # 5. 损失函数计算
#     #     # Loss = - [ Ratio * Advantage - Beta * KL ]
#     #     # 我们可以加入 PPO 的 Clip 机制来增加数值稳定性
        
#     #     surr1 = ratio * advantages
#     #     surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * advantages
#     #     policy_loss = -torch.min(surr1, surr2).mean()
        
#     #     # KL 惩罚项 (防止策略更新过大)
#     #     kl_loss = self.beta * approx_kl.mean()
        
#     #     loss = policy_loss + kl_loss
        
#     #     # 6. 反向传播与优化
#     #     self.optimizer.zero_grad()
#     #     loss.backward()
#     #     # 梯度裁剪 (防止 GRU 梯度爆炸)
#     #     torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
#     #     self.optimizer.step()
        
#     #     return {
#     #         "loss": loss.item(),
#     #         "policy_loss": policy_loss.item(),
#     #         "kl": approx_kl.mean().item(),
#     #         "avg_reward_raw": reward_batch.mean().item(),
#     #         "avg_reward_safe": adjusted_rewards.mean().item(),
#     #         "safety_trigger_rate": mask.float().mean().item()
#     #     }


#     # ds_grpo.py 修改建议

#     def update(self, rollouts):
#         """
#         支持变长序列的 GRPO 更新逻辑
#         Args:
#             rollouts: List[Dict], 每个元素是一个环境的完整轨迹数据
#         """
#         # 存储所有轨迹的 Loss
#         policy_losses = []
#         kl_losses = []
        
#         # 遍历每一条轨迹分别计算
#         for traj in rollouts:
#             # 1. 获取数据 (无需 Stack，直接取)
#             # obs: [T, 24], action: [T, 2], reward: [T]
#             obs = traj['obs'].unsqueeze(0).to(self.device)     # [1, T, 24]
#             action = traj['action'].unsqueeze(0).to(self.device) # [1, T, 2]
#             reward = traj['reward'].unsqueeze(0).to(self.device) # [1, T]
#             old_log_prob = traj['log_prob'].unsqueeze(0).to(self.device) # [1, T]
#             hidden = traj['hidden'].to(self.device) # [1, 1, H]

#             # 2. 安全过滤 (支持单条序列)
#             # 注意：safety_filter 需要适配 [1, T, ...] 维度
#             obs_for_safety = obs.clone()
#             # obs_for_safety[:, :, 8:] *= 30.0 # 根据之前的修复决定是否需要
            
#             adj_reward, mask = self.safety_filter(obs_for_safety, action, reward)
            
#             # 3. 计算优势 (Advantage) - 单条轨迹内部归一化? 
#             # GRPO 通常是组间归一化 (Group Normalization)。
#             # 如果我们是一个一个算，没法做 Group Norm。
#             # 因此，我们需要先收集所有轨迹的 Reward，算好 Mean/Std，再回来算 Loss。
#             pass 
        
#         # --- [修正版：先收集 Reward，再批量计算] ---
        
#         # 1. 预处理所有奖励
#         all_adj_rewards = []
#         all_masks = []
        
#         for traj in rollouts:
#             obs = traj['obs'].unsqueeze(0).to(self.device)
#             action = traj['action'].unsqueeze(0).to(self.device)
#             reward = traj['reward'].unsqueeze(0).to(self.device)
            
#             adj_r, mask = self.safety_filter(obs, action, reward)
#             all_adj_rewards.append(adj_r) # List of [1, T]
#             all_masks.append(mask)

#         # 2. 计算 Group Statistics
#         # 将所有时间步的奖励展平来计算均值方差 (Global Baseline)
#         # 或者计算每条轨迹的总分 (Episode Return) 来做对比
#         # 对于路径规划，通常每一步的 Advantage 更有意义
#         flat_rewards = torch.cat([r.flatten() for r in all_adj_rewards])
#         mean_r = flat_rewards.mean()
#         std_r = flat_rewards.std() + 1e-8
        
#         # 3. 计算 Loss
#         total_policy_loss = 0
#         total_kl = 0
#         total_steps = 0
        
#         for i, traj in enumerate(rollouts):
#             obs = traj['obs'].unsqueeze(0).to(self.device)
#             action = traj['action'].unsqueeze(0).to(self.device)
#             old_log_prob = traj['log_prob'].unsqueeze(0).to(self.device)
#             hidden = traj['hidden'].to(self.device)
            
#             # 归一化优势
#             adv = (all_adj_rewards[i] - mean_r) / std_r
            
#             # 网络前向
#             mu, std, _ = self.policy(obs, hidden)
#             dist = Normal(mu, std)
#             new_log_prob = dist.log_prob(action).sum(dim=-1)
            
#             # 计算 Ratio 和 KL
#             ratio = torch.exp(new_log_prob - old_log_prob)
#             approx_kl = old_log_prob - new_log_prob
            
#             surr1 = ratio * adv
#             surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * adv
            
#             # Sum over time steps
#             traj_loss = -torch.min(surr1, surr2).sum()
#             traj_kl = approx_kl.sum()
            
#             total_policy_loss += traj_loss
#             total_kl += traj_kl
#             total_steps += obs.shape[1]
            
#         # 平均 Loss
#         final_loss = (total_policy_loss + self.beta * total_kl) / total_steps
        
#         # 4. 优化
#         self.optimizer.zero_grad()
#         final_loss.backward()
#         torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
#         self.optimizer.step()
        
#         return {
#             "loss": final_loss.item(),
#             "avg_reward_safe": mean_r.item(),
#             # ... 其他 info
#         }

#     def save(self, path):
#         torch.save(self.policy.state_dict(), path)

#     def load(self, path):
#         self.policy.load_state_dict(torch.load(path, map_location=self.device))

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np

class DSGRPO:
    """
    DS-GRPO 算法核心类 (Enhanced Version)
    (Dynamic Safety-aware Group Relative Policy Optimization)
    
    包含:
    1. 策略网络管理 (Actor)
    2. 动态采样逻辑 (DAPO Idea)
    3. 组内安全过滤 (Safety Filtering)
    4. GRPO 损失函数计算与更新 (支持 Entropy Bonus 和 变长序列)
    """
    def __init__(self, 
                 policy_net, 
                 optimizer_lr=3e-4, 
                 beta_kl=0.01, 
                 clip_ratio=0.2,
                 entropy_coef=0.05, # [新增] 熵正则系数，用于强制探索
                 device='cuda'):
        """
        初始化算法
        
        Args:
            policy_net (nn.Module): 实例化的 RobustGRPOPolicy 网络
            optimizer_lr (float): 学习率
            beta_kl (float): KL 散度惩罚系数 (限制更新步幅)
            clip_ratio (float): PPO 风格的截断范围
            entropy_coef (float): 熵奖励系数 (关键！防止策略在初期坍缩为由确定的原地不动)
            device (str): 计算设备
        """
        self.policy = policy_net.to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=optimizer_lr)
        
        self.beta = beta_kl
        self.clip_ratio = clip_ratio
        self.entropy_coef = entropy_coef
        self.device = device
        
        # 物理常数 (用于安全过滤)
        self.dt = 0.5           # 环境时间步长
        self.agent_radius = 1.0 # 智能体半径估计

    def select_action(self, obs, hidden_state, deterministic=False):
        """
        选择动作 (推理阶段)
        
        Returns:
            action: 实际执行的动作 (未缩放)
            action_log_prob: 动作的对数概率
            next_hidden: 更新后的 GRU 隐状态
            mean_entropy: 平均策略熵 (用于 DAPO 和 监控)
        """
        with torch.no_grad():
            # 确保输入维度正确 [Batch, 24] -> [Batch, 1, 24]
            # 适配 GRU 的 (Batch, Seq, Feature) 输入要求
            if obs.dim() == 1:
                obs = obs.unsqueeze(0).unsqueeze(0)
            elif obs.dim() == 2:
                obs = obs.unsqueeze(1)
            
            obs = obs.to(self.device)
            if hidden_state is not None:
                hidden_state = hidden_state.to(self.device)

            # 网络前向传播
            mu, std, next_hidden = self.policy(obs, hidden_state)
            
            # 只有最后一帧用于决策
            mu = mu[:, -1, :]
            std = std[:, -1, :]
            
            dist = Normal(mu, std)
            
            if deterministic:
                action = mu
            else:
                action = dist.sample()
                
            # 计算 Log Probability 和 Entropy
            action_log_prob = dist.log_prob(action).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1)
            
            # 计算平均熵 (Scalar)，用于外部记录和 DAPO 逻辑
            mean_entropy = entropy.mean().item()
            
            return action.cpu().numpy(), action_log_prob.cpu().numpy(), next_hidden, mean_entropy

    def determine_group_size(self, uncertainty_metric, g_min=4, g_max=32):
        """
        [DAPO 核心] 根据不确定性动态决定采样组大小
        """
        # 简单的线性映射逻辑
        # 假设熵的范围大约在 0.0 ~ 3.0 之间
        normalized_u = np.clip(uncertainty_metric / 3.0, 0.0, 1.0)
        group_size = int(g_min + (g_max - g_min) * normalized_u)
        return group_size

    def safety_filter(self, obs_batch, action_batch, rewards_batch):
        """
        [Safety Filter 核心] 基于物理模型的组内安全过滤
        """
        # --- 1. 数据提取 ---
        # 提取雷达数据 [Batch, Seq, 16]
        # 注意: 假设外部传入的是真实物理距离 (0-30m)
        lidar_data = obs_batch[:, :, 8:] 
        
        # 动作反归一化 (Mapping [-1, 1] -> [-1, 2])
        # 必须与 Environment Wrapper 保持一致
        v_scaled = action_batch[:, :, 0]
        v_real = (v_scaled + 1.0) / 2.0 * 3.0 - 1.0
        
        # --- 2. 物理前瞻预测 ---
        # 获取每个时间步最近的障碍物距离
        min_lidar, _ = torch.min(lidar_data, dim=2) 
        
        # 预测行驶距离
        pred_dist = v_real * self.dt
        
        # --- 3. 危险判定 (微调版) ---
        # [优化] 减小安全余量 (0.8倍)，允许稍微靠近障碍物
        safe_margin = self.agent_radius * 0.8
        
        # 条件: (预测距离 > 实际距离 - 余量) AND (速度 > 0.3)
        # 只有真正快速冲向障碍物才罚，低速蠕动不罚
        is_dangerous = (pred_dist > (min_lidar - safe_margin)) & (v_real > 0.3)
        
        # --- 4. 奖励修正 ---
        adjusted_rewards = rewards_batch.clone()
        
        # [优化] 降低惩罚力度 (-50 -> -5.0)
        # 避免梯度过大导致网络崩溃，只要比正常负分大一点即可
        penalty = -5.0 
        
        adjusted_rewards[is_dangerous] += penalty
        
        return adjusted_rewards, is_dangerous

    def update(self, rollouts):
        """
        支持变长序列的 GRPO 更新逻辑 (Enhanced)
        
        Args:
            rollouts: List[Dict], 每个元素是一个环境的完整轨迹数据
                     (obs, action, reward, log_prob, hidden)
        """
        # --- 1. 预处理与安全过滤 ---
        
        all_adj_rewards = []
        all_masks = []
        
        for traj in rollouts:
            # 取出单条轨迹数据并增加 Batch 维度 [1, T, ...]
            obs = traj['obs'].unsqueeze(0).to(self.device)
            action = traj['action'].unsqueeze(0).to(self.device)
            reward = traj['reward'].unsqueeze(0).to(self.device)
            
            # 应用安全过滤器
            adj_r, mask = self.safety_filter(obs, action, reward)
            
            all_adj_rewards.append(adj_r) # List of [1, T]
            all_masks.append(mask)

        # --- 2. 计算 Group Statistics (Baseline) ---
        
        # 将所有轨迹的所有时间步奖励展平，计算全局均值和方差
        # 这是 GRPO "Relative" 的核心：你的表现相对于平均水平如何？
        flat_rewards = torch.cat([r.flatten() for r in all_adj_rewards])
        
        mean_r = flat_rewards.mean()
        std_r = flat_rewards.std()
        
        # [关键修正] 防止方差过小导致除零错误
        # 如果所有环境表现一致(std很小)，则不进行剧烈更新，或设为1防止爆炸
        if std_r < 1e-5:
            std_r = 1.0 
            # 此时 Advantage 接近 0，主要靠 Entropy Loss 驱动更新
        
        # --- 3. 计算 Loss 并反向传播 ---
        
        total_policy_loss = 0
        total_kl = 0
        total_entropy_loss = 0
        total_steps = 0
        
        # 遍历每条轨迹计算 Loss
        for i, traj in enumerate(rollouts):
            # 准备数据
            obs = traj['obs'].unsqueeze(0).to(self.device)
            action = traj['action'].unsqueeze(0).to(self.device)
            old_log_prob = traj['log_prob'].unsqueeze(0).to(self.device)
            hidden = traj['hidden'].to(self.device) # 初始隐状态
            
            # 计算归一化优势 Advantage
            # A = (R - Mean) / Std
            adv = (all_adj_rewards[i] - mean_r) / std_r
            
            # 网络重新前向传播 (Re-evaluate)
            # 必须传入初始 hidden，让 GRU 重新跑一遍整个序列
            mu, std, _ = self.policy(obs, hidden)
            dist = Normal(mu, std)
            
            # 计算新的 Log Prob 和 Entropy
            new_log_prob = dist.log_prob(action).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1)
            
            # 计算 Ratio 和 KL
            ratio = torch.exp(new_log_prob - old_log_prob)
            approx_kl = old_log_prob - new_log_prob
            
            # PPO-Clip Objective
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * adv
            
            # [关键新增] 熵正则项 (Entropy Bonus)
            # Loss = - (Policy_Objective + Entropy_Bonus - KL_Penalty)
            # 也就是: Minimize ( -Policy - Entropy + KL )
            
            traj_policy_loss = -torch.min(surr1, surr2).sum()
            traj_entropy_loss = -self.entropy_coef * entropy.sum() # 负号因为我们要最大化熵
            traj_kl_loss = self.beta * approx_kl.sum()
            
            # 累加 Loss
            total_policy_loss += traj_policy_loss
            total_entropy_loss += traj_entropy_loss
            total_kl += approx_kl.sum() # 记录原始 KL 用于监控
            total_steps += obs.shape[1]
            
        # 平均 Loss (Per Step)
        final_loss = (total_policy_loss + total_entropy_loss) / total_steps
        
        # 如果有 KL 惩罚，也可以加进去 (通常 GRPO 用 KL 提前停止，这里简化为 Loss 项)
        # final_loss += self.beta * (total_kl / total_steps)
        
        # --- 4. 优化步 ---
        self.optimizer.zero_grad()
        final_loss.backward()
        
        # 梯度裁剪 (防止 GRU 梯度爆炸)
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
        
        self.optimizer.step()
        
        # 先把每个 mask 拍扁成 [T_i]，然后再拼接成一个超长的一维向量 [Sum(T_i)]
        all_masks_flat = [m.flatten() for m in all_masks]

        return {
            "loss": final_loss.item(),
            "avg_reward_safe": mean_r.item(),
            "policy_loss": (total_policy_loss / total_steps).item(),
            "entropy_loss": (total_entropy_loss / total_steps).item(),
            "kl": (total_kl / total_steps).item(),
            
            # [修正] 使用 flatten 后的列表进行拼接
            "safety_trigger_rate": torch.cat(all_masks_flat).float().mean().item()
        }

    def save(self, path):
        torch.save(self.policy.state_dict(), path)

    def load(self, path):
        self.policy.load_state_dict(torch.load(path, map_location=self.device))