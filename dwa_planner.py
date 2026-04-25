import numpy as np
import math
import time
from uncertain_env import UncertainComplexEnv

class DWAConfig:
    """
    DWA 算法参数配置类
    """
    def __init__(self):
        # --- 机器人物理限制 ---
        self.max_speed = 2.0        # [m/s] 最大线速度 (对应环境 action space)
        self.min_speed = 0.0        # [m/s] 最小线速度 (DWA通常假设向前开，虽环境允许倒车，但规划通常只向前)
        self.max_yaw_rate = 1.0     # [rad/s] 最大角速度
        self.max_accel = 2.0        # [m/ss] 最大线加速度
        self.max_delta_yaw_rate = 2.0 # [rad/ss] 最大角加速度

        # --- 分辨率参数 (采样密度) ---
        self.v_reso = 0.1           # [m/s] 速度采样分辨率
        self.yaw_rate_reso = 0.1    # [rad/s] 角速度采样分辨率
        self.dt = 0.1               # [s] 轨迹预测的时间步长 (比环境的0.5s要细，为了更准)
        self.predict_time = 3.0     # [s] 向前预测的总时长

        # --- 评价函数权重 (核心参数) ---
        self.to_goal_cost_gain = 0.15   # 目标导向权重 (越大越想去终点)
        self.speed_cost_gain = 1.0      # 速度权重 (越大越想开得快)
        self.obstacle_cost_gain = 1.0   # 避障权重 (越大越怕撞)
        
        # --- 机器人尺寸 ---
        self.robot_stuck_flag_cons = 0.001  # 判断是否卡住的阈值
        self.robot_radius = 2.0             # [m] 机器人半径 (对应环境 agent_size)

class DWAPlanner:
    """
    DWA 局部规划器
    """
    def __init__(self, config):
        self.config = config
        
    def get_action(self, obs):
        """
        主接口：根据观测计算最佳动作 (v, w)
        :param obs: 环境返回的 24维 np.array
        """
        # 1. 解析观测数据 (State Estimation)
        # 即使观测有噪声，DWA 也只能相信它
        x = obs[0]
        y = obs[1]
        
        # 解析朝向 (cos, sin) -> angle
        # obs[2]=cos, obs[3]=sin
        yaw = math.atan2(obs[3], obs[2])
        
        target = obs[4:6] # [tx, ty]
        
        # 当前状态 state: [x, y, yaw, v, w]
        # 注意：observation 里没有直接给当前速度 v 和 w，
        # 在真实工程中需要从里程计获取。这里简化处理，假设当前速度为 0 或者上一帧的指令。
        # 为了闭环，我们在类外部维护一个 current_speed，或者每次假设从 0 开始搜索(比较激进)。
        # 更好的做法是让 get_action 接收 current_v, current_w。
        # 这里为了适配 gym 接口，我们做个简化：假设当前速度就是上次的规划速度。
        if not hasattr(self, 'current_v'):
            self.current_v = 0.0
            self.current_w = 0.0
            
        state = np.array([x, y, yaw, self.current_v, self.current_w])
        
        # 2. 解析障碍物 (Perception)
        # 将 Lidar 数据 (16维) 转换为局部点云坐标
        obstacles = self._process_lidar(obs[8:], x, y, yaw)
        
        # 3. 运行 DWA 核心计算
        u, trajectory = self._dwa_control(state, target, obstacles)
        
        # 更新内部状态
        self.current_v = u[0]
        self.current_w = u[1]
        
        return u, trajectory

    def _process_lidar(self, lidar_data, x, y, yaw):
        """将雷达距离数据转换为世界坐标系下的障碍物点集"""
        ob = []
        num_rays = len(lidar_data)
        angle_inc = (2 * np.pi) / num_rays
        max_range = 30.0 # 环境定义的雷达最大距离
        
        for i in range(num_rays):
            dist = lidar_data[i]
            # 过滤掉达到最大量程的读数 (视为空旷)
            if dist < max_range - 0.5: 
                # 计算该射线的绝对角度
                angle = yaw + i * angle_inc
                # 转换为世界坐标
                ox = x + dist * np.cos(angle)
                oy = y + dist * np.sin(angle)
                ob.append([ox, oy])
        
        return np.array(ob)

    def _dwa_control(self, state, goal, ob):
        """DWA 核心控制循环"""
        # 1. 计算动态窗口 (Dynamic Window)
        dw = self._calc_dynamic_window(state)
        
        # 2. 并在窗口内搜索最佳控制量 (u) 和对应的轨迹
        u, trajectory = self._calc_control_and_trajectory(state, dw, goal, ob)
        
        return u, trajectory

    def _calc_dynamic_window(self, state):
        """
        计算速度搜索范围
        window = [v_min, v_max, w_min, w_max]
        """
        # 1. 车辆物理极限
        Vs = [self.config.min_speed, self.config.max_speed,
              -self.config.max_yaw_rate, self.config.max_yaw_rate]

        # 2. 运动学极限 (根据当前速度和最大加速度能达到的范围)
        # dt 通常取环境的 step 时间或者 DWA 的预测周期，这里取环境的 0.5s 可能太大，
        # 为了保证控制的平滑性，我们用 config.dt * 5 (假设 0.5s 内能加速这么多)
        # 或者严谨点：DWA假设在下一个控制周期内能达到的速度
        sim_dt = 0.5 # 环境 Step 时间
        
        Vd = [state[3] - self.config.max_accel * sim_dt,
              state[3] + self.config.max_accel * sim_dt,
              state[4] - self.config.max_delta_yaw_rate * sim_dt,
              state[4] + self.config.max_delta_yaw_rate * sim_dt]

        # 3. 取交集
        dw = [max(Vs[0], Vd[0]), min(Vs[1], Vd[1]),
              max(Vs[2], Vd[2]), min(Vs[3], Vd[3])]

        return dw

    def _calc_control_and_trajectory(self, state, dw, goal, ob):
        """
        遍历窗口内的所有速度组合，模拟轨迹并打分
        """
        x_init = state[:]
        min_cost = float("inf")
        best_u = [0.0, 0.0]
        best_trajectory = np.array([x_init])

        # 遍历线速度 v
        for v in np.arange(dw[0], dw[1], self.config.v_reso):
            # 遍历角速度 w
            for w in np.arange(dw[2], dw[3], self.config.yaw_rate_reso):
                
                # 1. 预测轨迹
                trajectory = self._predict_trajectory(x_init, v, w)
                
                # 2. 计算各项代价
                # 目标代价 (Heading Cost)
                to_goal_cost = self.config.to_goal_cost_gain * self._calc_to_goal_cost(trajectory, goal)
                
                # 速度代价 (Speed Cost) - 希望越快越好，所以取倒数或负数，这里用 max_v - v
                speed_cost = self.config.speed_cost_gain * (self.config.max_speed - trajectory[-1, 3])
                
                # 避障代价 (Obstacle Cost)
                ob_cost = self.config.obstacle_cost_gain * self._calc_obstacle_cost(trajectory, ob)

                # 总代价
                final_cost = to_goal_cost + speed_cost + ob_cost

                # 3. 寻找最小代价
                if final_cost < min_cost:
                    min_cost = final_cost
                    best_u = [v, w]
                    best_trajectory = trajectory
                    
                    # 强硬约束：如果预测会撞，坚决不选 (代价设为无限)
                    if ob_cost == float("inf"):
                        min_cost = float("inf") # 回滚

        return best_u, best_trajectory

    def _predict_trajectory(self, x_init, v, w):
        """
        基于运动学模型 (x, y, theta, v, w) 预测未来一段时间的轨迹
        """
        state = np.array(x_init)
        trajectory = np.array(state)
        time_steps = 0
        
        while time_steps < self.config.predict_time:
            # 简单的差分驱动模型
            # x += v * cos(theta) * dt
            # y += v * sin(theta) * dt
            # theta += w * dt
            state[0] += v * math.cos(state[2]) * self.config.dt
            state[1] += v * math.sin(state[2]) * self.config.dt
            state[2] += w * self.config.dt
            state[3] = v
            state[4] = w

            trajectory = np.vstack((trajectory, state))
            time_steps += self.config.dt

        return trajectory

    def _calc_to_goal_cost(self, trajectory, goal):
        """
        计算轨迹末端到目标的偏差
        """
        # 取预测轨迹的最后一个点
        dx = goal[0] - trajectory[-1, 0]
        dy = goal[1] - trajectory[-1, 1]
        error_angle = math.atan2(dy, dx)
        cost_angle = error_angle - trajectory[-1, 2]
        cost = abs(math.atan2(math.sin(cost_angle), math.cos(cost_angle)))
        
        # 也可以加入距离代价
        dist = math.hypot(dx, dy)
        
        # 这里的 cost 主要由角度偏差决定，距离作为辅助
        return cost + 0.1 * dist

    def _calc_obstacle_cost(self, trajectory, ob):
        """
        计算轨迹上所有点到最近障碍物的距离
        """
        if len(ob) == 0:
            return 0.0

        min_r = float("inf")

        # 遍历轨迹上的每一个点
        for i in range(len(trajectory)):
            x_traj = trajectory[i, 0]
            y_traj = trajectory[i, 1]
            
            # 找到离这个点最近的障碍物
            # (这里可以用 KDTree 加速，但对于只有16个雷达点的小数据量，暴力算也很快)
            dx = x_traj - ob[:, 0]
            dy = y_traj - ob[:, 1]
            r = np.hypot(dx, dy)
            
            min_r_step = np.min(r)
            if min_r_step < min_r:
                min_r = min_r_step

        # 判定碰撞
        if min_r <= self.config.robot_radius:
            return float("inf") # 碰撞！代价无穷大

        # 距离越近，代价越高；距离越远，代价越低 (取倒数)
        return 1.0 / min_r

# ==========================================
# 主运行脚本 (用于测试 DWA)
# ==========================================
if __name__ == "__main__":
    # 1. 初始化环境
    # 设置一个起点和终点
    start_pos = [10.0, 10.0]
    target_pos = [90.0, 90.0]
    
    env = UncertainComplexEnv(render_mode="human", start_pos=start_pos, target_pos=target_pos)
    
    # 2. 初始化 DWA 规划器
    config = DWAConfig()
    dwa = DWAPlanner(config)
    
    obs, _ = env.reset()
    done = False
    truncated = False
    
    print("Start DWA Navigation...")
    
    step = 0
    max_steps = 1000
    
    total_reward = 0
    
    while not done and not truncated and step < max_steps:
        env.render()
        
        # 3. 获取 DWA 决策
        # DWA 内部会处理 Lidar 数据并避障
        action, pred_traj = dwa.get_action(obs)
        
        # 4. 执行动作
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        step += 1
        
        # 简单的进度打印
        if step % 10 == 0:
            dist = np.linalg.norm(env.agent_pos - env.target_pos)
            print(f"Step: {step}, Action: v={action[0]:.2f}, w={action[1]:.2f}, Dist to Goal: {dist:.1f}")

    if done:
        print(f"Goal Reached! Total Reward: {total_reward:.1f}")
    else:
        print("Failed or Max Steps Reached.")
        
    env.close()