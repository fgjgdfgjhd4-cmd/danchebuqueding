import numpy as np
import math
import time
from scipy.optimize import minimize
from uncertain_env import UncertainComplexEnv

class MPCConfig:
    """
    MPC 参数配置类
    """
    def __init__(self):
        # --- 预测参数 ---
        self.horizon = 10           # [步] 预测视界长度 (预测未来多少步)
        self.dt = 0.2               # [s] 预测的时间步长 (为了精度，比环境的0.5s小)
        
        # --- 物理约束 (对应环境) ---
        self.v_min = -1.0
        self.v_max = 2.0
        self.w_min = -1.0
        self.w_max = 1.0
        self.max_accel = 2.0        # 假设的加速度限制，用于平滑控制
        self.max_dyaw = 2.0         # 假设的角加速度限制
        
        # --- 代价权重 (Cost Weights) ---
        # 目标函数 J = w_g * dist_goal + w_h * heading_err + w_u * control_effort + w_obs * obstacle
        self.w_goal = 2.0           # 目标距离权重
        self.w_heading = 0.5        # 航向偏差权重
        self.w_action = 0.1         # 动作幅度权重 (省电/平滑)
        self.w_smooth = 0.5         # 动作变化率权重 (平滑)
        self.w_obs = 100.0          # 避障权重 (软约束)
        
        # --- 安全参数 ---
        self.robot_radius = 2.0     # 机器人半径
        self.safe_margin = 0.5      # 安全余量

class MPCPlanner:
    """
    基于 Scipy 的非线性模型预测控制器
    """
    def __init__(self, config):
        self.config = config
        
        # 动作维度 (v, w)
        self.u_dim = 2 
        # 优化变量的总长度 = horizon * u_dim
        self.sol_dim = self.config.horizon * self.u_dim
        
        # 缓存上一次的解，作为热启动 (Warm Start)
        self.prev_sol = np.zeros(self.sol_dim)
        
        # 定义动作边界
        # bounds 格式: [(v_min, v_max), (w_min, w_max), ...] 重复 N 次
        self.bounds = []
        for _ in range(self.config.horizon):
            self.bounds.append((self.config.v_min, self.config.v_max))
            self.bounds.append((self.config.w_min, self.config.w_max))

    def get_action(self, obs):
        """
        主接口：接收环境观测，返回 MPC 计算出的最优动作
        """
        # 1. 状态解析
        x, y = obs[0], obs[1]
        yaw = math.atan2(obs[3], obs[2]) # cos, sin -> yaw
        target_pos = obs[4:6]
        
        # 当前状态向量
        current_state = np.array([x, y, yaw])
        
        # 2. 感知处理：将 Lidar 数据转为障碍物坐标
        # MPC 需要知道障碍物在世界坐标系的确切位置
        obstacles = self._process_lidar(obs[8:], x, y, yaw)
        
        # 3. 求解优化问题
        # 目标：找到 u_seq = [u_0, u_1, ..., u_N-1] 最小化 Cost
        
        # 热启动：将上次的解左移一步，末尾补零
        # 这样优化器能更快收敛
        u_init = np.roll(self.prev_sol, -self.u_dim)
        u_init[-self.u_dim:] = 0.0
        
        # 调用 Scipy 优化器 (SLSQP 适合带约束的非线性优化)
        # 注意：Python 的 minimize 比较慢，为了实时性，我们减少迭代次数或精度
        res = minimize(
            fun=self._cost_function,
            x0=u_init,
            args=(current_state, target_pos, obstacles),
            method='SLSQP',
            bounds=self.bounds,
            options={'ftol': 1e-3, 'disp': False, 'maxiter': 20} 
        )
        
        # 4. 获取结果
        self.prev_sol = res.x # 更新缓存
        
        # 取出预测序列中的第一个动作作为当前执行动作
        best_u = res.x[:2] # [v0, w0]
        
        # 可选：返回预测轨迹用于可视化
        pred_traj = self._predict_trajectory(current_state, res.x)
        
        return best_u, pred_traj

    def _process_lidar(self, lidar_data, x, y, yaw):
        """将雷达数据转换为障碍物点云"""
        obs_points = []
        num_rays = len(lidar_data)
        angle_inc = (2 * np.pi) / num_rays
        max_range = 30.0
        
        for i in range(num_rays):
            dist = lidar_data[i]
            if dist < max_range - 0.5: # 有效障碍物
                angle = yaw + i * angle_inc
                ox = x + dist * np.cos(angle)
                oy = y + dist * np.sin(angle)
                obs_points.append([ox, oy])
                
        return np.array(obs_points)

    def _predict_trajectory(self, x0, u_seq):
        """
        根据给定的控制序列推演未来轨迹
        x0: [x, y, theta]
        u_seq: [v0, w0, v1, w1, ...]
        return: 轨迹点数组 (N+1, 3)
        """
        traj = [x0]
        curr_x = x0.copy()
        
        # u_seq 展平了，需重新 reshape
        controls = u_seq.reshape(self.config.horizon, self.u_dim)
        
        for u in controls:
            v, w = u[0], u[1]
            dt = self.config.dt
            
            # 运动学模型 (Differential Drive)
            # x_{t+1} = x_t + v * cos(theta) * dt
            # y_{t+1} = y_t + v * sin(theta) * dt
            # theta_{t+1} = theta_t + w * dt
            next_x = curr_x.copy()
            next_x[0] += v * np.cos(curr_x[2]) * dt
            next_x[1] += v * np.sin(curr_x[2]) * dt
            next_x[2] += w * dt
            
            traj.append(next_x)
            curr_x = next_x
            
        return np.array(traj)

    def _cost_function(self, u_flat, current_state, target_pos, obstacles):
        """
        代价函数 J
        """
        cost = 0.0
        
        # 1. 轨迹推演
        # 为了速度，这里不再调用 _predict_trajectory，而是直接手写循环计算 Cost
        # 这样减少函数调用开销
        curr_x = current_state.copy()
        controls = u_flat.reshape(self.config.horizon, self.u_dim)
        
        prev_u = np.zeros(2) # 用于计算动作平滑度
        
        for i, u in enumerate(controls):
            v, w = u[0], u[1]
            dt = self.config.dt
            
            # --- 状态更新 ---
            curr_x[0] += v * np.cos(curr_x[2]) * dt
            curr_x[1] += v * np.sin(curr_x[2]) * dt
            curr_x[2] += w * dt
            
            # --- 代价计算 ---
            
            # A. 目标距离代价
            dist_to_goal = np.hypot(curr_x[0] - target_pos[0], curr_x[1] - target_pos[1])
            cost += self.config.w_goal * dist_to_goal
            
            # B. 航向代价 (希望车头对准终点)
            # 只有当距离比较远时才强烈要求对准，防止终点附近震荡
            if dist_to_goal > 1.0:
                desired_yaw = math.atan2(target_pos[1] - curr_x[1], target_pos[0] - curr_x[0])
                yaw_err = abs(desired_yaw - curr_x[2])
                # 归一化角度误差到 [-pi, pi]
                yaw_err = math.atan2(math.sin(yaw_err), math.cos(yaw_err))
                cost += self.config.w_heading * abs(yaw_err)
            
            # C. 动作代价 (惩罚过大动作和倒车)
            cost += self.config.w_action * (v**2 + w**2)
            if v < 0: cost += 10.0 * abs(v) # 尽量不倒车
            
            # D. 平滑代价 (惩罚突变)
            if i > 0:
                du = u - prev_u
                cost += self.config.w_smooth * (du[0]**2 + du[1]**2)
            prev_u = u
            
            # E. 避障代价 (最耗时部分)
            # 只计算离机器人最近的障碍物
            if len(obstacles) > 0:
                # 向量化计算当前位置到所有障碍点的距离
                # obstacles: (M, 2), curr_pos: (2,)
                dists = np.hypot(obstacles[:, 0] - curr_x[0], obstacles[:, 1] - curr_x[1])
                min_dist = np.min(dists)
                
                # 软约束：如果距离小于安全半径，施加巨大的指数惩罚
                # barrier function
                safe_dist = self.config.robot_radius + self.config.safe_margin
                if min_dist < safe_dist:
                    cost += self.config.w_obs * (1.0 / (min_dist + 1e-4))
                    # 如果发生碰撞，加巨额惩罚
                    if min_dist < self.config.robot_radius:
                        cost += 1000.0

        return cost

# ==========================================
# 主运行脚本
# ==========================================
if __name__ == "__main__":
    # 1. 初始化环境
    start_pos = [10.0, 10.0]
    target_pos = [90.0, 90.0]
    
    env = UncertainComplexEnv(render_mode="human", start_pos=start_pos, target_pos=target_pos)
    
    # 2. 初始化 MPC 规划器
    config = MPCConfig()
    planner = MPCPlanner(config)
    
    obs, _ = env.reset()
    done = False
    
    print("Start MPC Navigation...")
    print("Note: MPC optimization in Python (scipy) can be slow (5-10Hz).")
    
    step = 0
    max_steps = 1000
    total_reward = 0
    
    try:
        while not done and step < max_steps:
            start_time = time.time()
            
            env.render()
            
            # 3. 获取 MPC 决策
            action, pred_traj = planner.get_action(obs)
            
            # 可选：将预测轨迹画出来 (如果环境支持绘制 debug 线条)
            # 这里简单略过，直接执行
            
            # 4. 执行动作
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            step += 1
            
            # 计算推理耗时
            solve_time = time.time() - start_time
            
            if step % 5 == 0:
                dist = np.linalg.norm(env.agent_pos - env.target_pos)
                print(f"Step: {step}, Action: v={action[0]:.2f}, w={action[1]:.2f}, Dist: {dist:.1f}, Time: {solve_time*1000:.1f}ms")
                
    except KeyboardInterrupt:
        print("Stopped by user.")
    finally:
        if done:
            print(f"Goal Reached! Total Reward: {total_reward:.1f}")
        else:
            print("Failed or Max Steps Reached.")
        env.close()