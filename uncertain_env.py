
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math
import pyglet
from pyglet import shapes
import time
import os

class DynamicObstacle:
    """
    动态障碍物类 (增强版)
    包含更严格的物理碰撞检测，防止穿墙和重叠
    """
    def __init__(self, id, pos, velocity, radius=1.5):
        self.id = id # 唯一标识符
        self.pos = np.array(pos, dtype=np.float32)
        self.velocity = np.array(velocity, dtype=np.float32) # [vx, vy]
        self.prev_pos = self.pos.copy()
        self.smoothed_velocity = self.velocity.copy()
        self.radius = radius

    def update(self, dt, map_width, map_height, map_matrix, other_obstacles):
        """
        更新位置，包含物理碰撞处理
        """
        # 1. 计算预期位置
        self.prev_pos = self.pos.copy()
        next_pos = self.pos + self.velocity * dt
        
        collision_detected = False
        
        # --- A. 边界检测 ---
        if (next_pos[0] < self.radius or next_pos[0] > map_width - self.radius or
            next_pos[1] < self.radius or next_pos[1] > map_height - self.radius):
            
            if next_pos[0] < self.radius or next_pos[0] > map_width - self.radius:
                self.velocity[0] *= -1
            if next_pos[1] < self.radius or next_pos[1] > map_height - self.radius:
                self.velocity[1] *= -1
            collision_detected = True

        # --- B. 静态障碍物检测 (墙壁) ---
        if not collision_detected:
            check_angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
            for angle in check_angles:
                check_x = next_pos[0] + np.cos(angle) * self.radius
                check_y = next_pos[1] + np.sin(angle) * self.radius
                ix, iy = int(check_x), int(check_y)
                if 0 <= ix < 100 and 0 <= iy < 100:
                    if map_matrix[ix, iy] == 1:
                        self.velocity *= -1 
                        collision_detected = True
                        break
        
        # --- C. 动态障碍物互斥 ---
        if not collision_detected:
            for other in other_obstacles:
                if other.id == self.id: continue 
                dist = np.linalg.norm(next_pos - other.pos)
                min_dist = self.radius + other.radius + 0.2 
                if dist < min_dist:
                    self.velocity *= -1
                    collision_detected = True
                    break

        # --- 3. 更新状态 ---
        if not collision_detected:
            self.pos = next_pos
            measured_velocity = (self.pos - self.prev_pos) / max(dt, 1e-6)
            self.smoothed_velocity = 0.7 * self.smoothed_velocity + 0.3 * measured_velocity
        else:
            self.smoothed_velocity = 0.8 * self.smoothed_velocity + 0.2 * self.velocity

class UncertainComplexEnv(gym.Env):
    """
    单车部分不确定性路径规划环境 (Reward Shaping 终极版)
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, map_matrix=None, render_mode=None, start_pos=None, target_pos=None):
        super(UncertainComplexEnv, self).__init__()
        
        self.render_mode = render_mode
        self.window = None
        self.batch = None

        # --- 1. 不确定性参数 ---
        self.noise_params = {
            "pos_std": 0.5,      
            "heading_std": 0.05, 
            "lidar_std": 0.2,    
        }
        
        self.num_dynamic_obs = 15 
        self.dynamic_obstacles = [] 

        # --- 2. 基础环境参数 ---
        self.width = 100.0
        self.height = 100.0
        self.agent_size = 2.0  
        self.dt = 0.5          
        
        if map_matrix is None:
            self.map_matrix = self._generate_complex_map()
        else:
            self.map_matrix = map_matrix
            
        # --- 3. 动作与观测空间 ---
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32), 
            high=np.array([2.0, 1.0], dtype=np.float32), 
            shape=(2,), dtype=np.float32
        )

        self.lidar_rays = 16
        self.lidar_range = 30.0
        
        # 观测空间定义 (24维)
        low_pos = np.array([0, 0], dtype=np.float32)
        high_pos = np.array([self.width, self.height], dtype=np.float32)
        low_head = np.array([-1, -1], dtype=np.float32)
        high_head = np.array([1, 1], dtype=np.float32)
        low_target = np.array([0, 0], dtype=np.float32)
        high_target = np.array([self.width, self.height], dtype=np.float32)
        max_dist = np.sqrt(self.width**2 + self.height**2)
        low_rel = np.array([0, -np.pi], dtype=np.float32)
        high_rel = np.array([max_dist, np.pi], dtype=np.float32)
        low_lidar = np.zeros(self.lidar_rays, dtype=np.float32)
        high_lidar = np.full(self.lidar_rays, self.lidar_range, dtype=np.float32)

        low_obs = np.concatenate([low_pos, low_head, low_target, low_rel, low_lidar])
        high_obs = np.concatenate([high_pos, high_head, high_target, high_rel, high_lidar])
        
        self.observation_space = spaces.Box(low=low_obs, high=high_obs, dtype=np.float32)

        self.agent_pos = np.zeros(2) 
        self.agent_dir = 0.0         
        self.target_pos = np.zeros(2)
        
        self.trajectory = []         
        self.planned_path = None
        self.dashed_lines_cache = []
        self.current_step = 0
        self.max_steps = 1000 
        
        self.default_start_pos = np.array(start_pos, dtype=np.float32) if start_pos is not None else None
        self.default_target_pos = np.array(target_pos, dtype=np.float32) if target_pos is not None else None

        self.last_action = None
        self.prediction_horizon_steps = 3
        self.prediction_decay = 0.75
        self.base_safety_margin = 1.5
        self.safety_penalty_gain = 1.25
        self.safety_bonus_gain = 0.08
        self.prediction_cache = []
        self.last_safety_metrics = {
            "uncertainty_penalty": 0.0,
            "risk_min_clearance": float("inf"),
            "risk_margin": 0.0,
        }

    def set_start_pos(self, pos):
        self.default_start_pos = np.array(pos, dtype=np.float32)

    def set_target_pos(self, pos):
        self.default_target_pos = np.array(pos, dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        start_pos = options.get('start_pos') if options else None
        target_pos = options.get('target_pos') if options else None
        
        if start_pos is None: start_pos = self.default_start_pos
        if target_pos is None: target_pos = self.default_target_pos
        
        if start_pos is not None:
            self.agent_pos = np.array(start_pos, dtype=np.float32)
        else:
            self.agent_pos = self._sample_free_pos()

        if target_pos is not None:
            self.target_pos = np.array(target_pos, dtype=np.float32)
        else:
            while True:
                candidate = self._sample_free_pos()
                if np.linalg.norm(self.agent_pos - candidate) > 30.0:
                    self.target_pos = candidate
                    break
        
        self.agent_dir = 0.0
        self.trajectory = [self.agent_pos.copy()]
        self.current_step = 0

        # # [新增] 初始化上一步动作记录，用于计算平滑度
        # self.last_action = np.array([0.0, 0.0], dtype=np.float32)
        # [新增] 初始化上一步动作记录，用于计算平滑度
        self.last_action = np.array([0.0, 0.0], dtype=np.float32)
        self.prediction_cache = []
        self.last_safety_metrics = {
            "uncertainty_penalty": 0.0,
            "risk_min_clearance": float("inf"),
            "risk_margin": 0.0,
        }

        # 初始化动态障碍物
        self.dynamic_obstacles = []
        for i in range(self.num_dynamic_obs):
            valid_spawn = False
            while not valid_spawn:
                spawn_x = self.np_random.uniform(0.0, self.width)
                spawn_y = self.np_random.uniform(0.0, self.height)
                pos = np.array([spawn_x, spawn_y], dtype=np.float32)
                
                if self._check_circle_collision_with_map(pos, radius=2.0): continue
                if np.linalg.norm(pos - self.agent_pos) < 15.0: continue
                
                overlap_other = False
                for existing_obs in self.dynamic_obstacles:
                    if np.linalg.norm(pos - existing_obs.pos) < 4.0:
                        overlap_other = True
                        break
                if overlap_other: continue
                
                valid_spawn = True
                vel = self.np_random.uniform(-0.8, 0.8, size=2)
                # vel = self.np_random.uniform(0, 0.5, size=2)

                self.dynamic_obstacles.append(DynamicObstacle(i, pos, vel))
        
        obs = self._get_obs()
        info = self._get_info()
        
        if self.render_mode == "human":
            self.render()
            
        return obs, info

    def step(self, action):
        self.current_step += 1
        
        v_cmd = np.clip(action[0], -1.0, 2.0)
        w_cmd = np.clip(action[1], -1.0, 1.0)
        
        # --- 1. 物理更新 ---
        self.agent_dir += w_cmd * self.dt
        self.agent_dir = (self.agent_dir + np.pi) % (2 * np.pi) - np.pi
        
        dx = v_cmd * np.cos(self.agent_dir) * self.dt
        dy = v_cmd * np.sin(self.agent_dir) * self.dt
        new_pos = self.agent_pos + np.array([dx, dy])
        
        for obs in self.dynamic_obstacles:
            obs.update(self.dt, self.width, self.height, self.map_matrix, self.dynamic_obstacles)
        
        terminated = False
        collision_static = self._check_static_collision(new_pos)
        collision_dynamic = self._check_dynamic_collision(new_pos)
        self.prediction_cache = self._predict_dynamic_obstacles()
        
        # --- 2. 奖励函数 (Ultimate Version) ---
        
        # A. 基础生存消耗
        reward = -0.01 

        dist_to_target = np.linalg.norm(new_pos - self.target_pos)
        prev_dist = np.linalg.norm(self.agent_pos - self.target_pos)
        
        # B. 进度奖励 (微分量：鼓励移动)
        progress = prev_dist - dist_to_target
        reward += progress * 2.0 
        
        # C. [关键新增] 距离吸引奖励 (状态量：鼓励靠近)
        # 这是一个非线性函数：离得越近，值越大。
        # 当距离=50m时，奖励约 0.1；当距离=5m时，奖励约 0.8；当距离=0m时，奖励5.0
        # 这会产生强大的"引力场"
        reward += 5.0 / (dist_to_target + 1.0)

        # D. 朝向奖励 (辅助引导)
        vec_to_target = self.target_pos - self.agent_pos
        target_angle = math.atan2(vec_to_target[1], vec_to_target[0])
        heading_error = target_angle - self.agent_dir
        heading_reward = np.cos(heading_error) 
        reward += 0.05 * heading_reward 

        # E. 动作惩罚 (Action Penalty)
        # 严厉禁止倒车
        if v_cmd < 0:
            reward -= 0.2
        # 平滑惩罚
        # reward -= 0.5 * abs(w_cmd)

        # =========================================================
        # [修改] 平滑度奖励 (Smoothness Penalty)
        # =========================================================
        # 计算当前动作与上一步动作的差值 (L2 范数)
        # action 是归一化后的[-1, 1]还是真实物理值都可以，建议用真实物理值
        current_action_vec = np.array([v_cmd, w_cmd])
        action_diff = np.linalg.norm(current_action_vec - self.last_action)
        
        # 系数 0.5 可调：越大路径越平滑，但反应越迟钝
        smoothness_penalty = 0.4 * action_diff
        reward -= smoothness_penalty
        
        # 更新记录
        self.last_action = current_action_vec

        # =========================================================
        # [新增] 防卡死惩罚 (Anti-Freezing)
        # =========================================================
        # 如果速度太慢（比如小于 0.1 m/s），给予额外惩罚
        # 这个惩罚应该足够大，让它觉得"不动"比"慢慢试探"更亏
        if v_cmd < 0.1:
            reward -= 0.5  # 每一步扣 0.5，相当于 10秒不动就扣 50分
            
            # (可选) 甚至可以更激进：检测到长时间不动直接判负结束
            # if self.low_speed_timer > 50: terminated = True
        # =========================================================

        # 状态标记 (新增)
        # 状态标记 (新增)
        uncertainty_penalty, risk_min_clearance, risk_margin = self._compute_uncertainty_aware_safety_reward(new_pos)
        reward -= uncertainty_penalty
        self.last_safety_metrics = {
            "uncertainty_penalty": float(uncertainty_penalty),
            "risk_min_clearance": float(risk_min_clearance),
            "risk_margin": float(risk_margin),
        }
        if risk_min_clearance > risk_margin + 1.0:
            reward += self.safety_bonus_gain
        is_success = False
        is_collision = False

        # F. 碰撞与结束奖励
        if collision_static or collision_dynamic:
            # [关键修改] 加大碰撞惩罚 (从 -20 加大到 -50)
            # 因为我们增加了正向奖励，必须同步增加负向惩罚以保持平衡
            reward -= 50.0
            terminated = True
            is_collision = True
        elif dist_to_target < 2.0:
            # 到达奖励
            reward += 200.0
            terminated = True
            self.agent_pos = new_pos 
            is_success = True
        else:
            self.agent_pos = new_pos 
            
        self.trajectory.append(self.agent_pos.copy())
        
        truncated = False
        if self.current_step >= self.max_steps:
            truncated = True
            
        obs = self._get_obs()
        info = self._get_info()
        info["is_success"] = is_success
        info["is_collision"] = is_collision
        
        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        # 1. 定位噪声
        pos_noise = self.np_random.normal(0, self.noise_params["pos_std"], size=2)
        obs_pos = self.agent_pos + pos_noise
        
        heading_noise = self.np_random.normal(0, self.noise_params["heading_std"])
        obs_heading_rad = self.agent_dir + heading_noise
        
        heading_vec = np.array([np.cos(obs_heading_rad), np.sin(obs_heading_rad)])
        
        vec_to_target = self.target_pos - obs_pos
        dist = np.linalg.norm(vec_to_target)
        angle_to_target = math.atan2(vec_to_target[1], vec_to_target[0]) - obs_heading_rad
        angle_to_target = (angle_to_target + np.pi) % (2 * np.pi) - np.pi
        
        # 2. 激光雷达噪声
        clean_lidar = self._simulate_lidar_with_dynamic()
        noisy_lidar = clean_lidar + self.np_random.normal(0, self.noise_params["lidar_std"], size=self.lidar_rays)
        noisy_lidar = np.clip(noisy_lidar, 0, self.lidar_range)
        
        obs = np.concatenate([
            obs_pos,      
            heading_vec,  
            self.target_pos, 
            [dist, angle_to_target],
            noisy_lidar   
        ]).astype(np.float32)
        
        return obs

    def _simulate_lidar_with_dynamic(self):
        readings = []
        angle_inc = (2 * np.pi) / self.lidar_rays
        
        for i in range(self.lidar_rays):
            angle = self.agent_dir + i * angle_inc
            
            # 1. 静态检测
            static_dist = self.lidar_range
            dist = 0
            step = 1.0
            cx, cy = self.agent_pos
            vx, vy = np.cos(angle), np.sin(angle)
            
            while dist < self.lidar_range:
                dist += step
                cx += vx * step
                cy += vy * step
                
                if cx < 0 or cx >= self.width or cy < 0 or cy >= self.height:
                    static_dist = dist
                    break
                if self.map_matrix[int(cx), int(cy)] == 1:
                    static_dist = dist
                    break
            else:
                static_dist = self.lidar_range

            # 2. 动态检测
            min_dynamic_dist = self.lidar_range
            for obs in self.dynamic_obstacles:
                f = self.agent_pos - obs.pos
                a = 1.0 
                b = 2 * np.dot(f, np.array([vx, vy]))
                c = np.dot(f, f) - obs.radius**2
                
                discriminant = b*b - 4*a*c
                if discriminant >= 0:
                    t1 = (-b - np.sqrt(discriminant)) / (2*a)
                    t2 = (-b + np.sqrt(discriminant)) / (2*a)
                    if 0 < t1 < min_dynamic_dist: min_dynamic_dist = t1
                    if 0 < t2 < min_dynamic_dist: min_dynamic_dist = t2
            
            readings.append(min(static_dist, min_dynamic_dist))
            
        return np.array(readings)

    def _check_static_collision(self, pos):
        if pos[0] < 0 or pos[0] >= self.width or pos[1] < 0 or pos[1] >= self.height:
            return True
        r = self.agent_size / 2.0
        check_points = [pos, pos+[r,r], pos+[r,-r], pos+[-r,r], pos+[-r,-r]]
        for p in check_points:
            ix, iy = int(p[0]), int(p[1])
            ix, iy = np.clip(ix, 0, 99), np.clip(iy, 0, 99)
            if self.map_matrix[ix, iy] == 1:
                return True
        return False

    def _check_circle_collision_with_map(self, center, radius):
        angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
        for ang in angles:
            cx = center[0] + np.cos(ang) * radius
            cy = center[1] + np.sin(ang) * radius
            ix, iy = int(cx), int(cy)
            if ix < 0 or ix >= 100 or iy < 0 or iy >= 100: return True
            if self.map_matrix[ix, iy] == 1: return True
        ix, iy = int(center[0]), int(center[1])
        if self.map_matrix[ix, iy] == 1: return True
        return False

    def _check_dynamic_collision(self, pos):
        agent_radius = self.agent_size / 2.0
        for obs in self.dynamic_obstacles:
            dist = np.linalg.norm(pos - obs.pos)
            if dist < (agent_radius + obs.radius):
                return True
        return False
    
    def _predict_dynamic_obstacles(self):
        predictions = []
        for obs in self.dynamic_obstacles:
            future_positions = []
            future_uncertainties = []
            predicted_pos = obs.pos.copy()
            velocity = getattr(obs, "smoothed_velocity", obs.velocity)
            speed = np.linalg.norm(velocity)

            for step_idx in range(1, self.prediction_horizon_steps + 1):
                predicted_pos = predicted_pos + velocity * self.dt
                predicted_pos = np.clip(
                    predicted_pos,
                    [obs.radius, obs.radius],
                    [self.width - obs.radius, self.height - obs.radius],
                )
                uncertainty = self._estimate_prediction_uncertainty(step_idx, speed)
                future_positions.append(predicted_pos.copy())
                future_uncertainties.append(float(uncertainty))

            predictions.append(
                {
                    "id": obs.id,
                    "radius": obs.radius,
                    "speed": float(speed),
                    "future_positions": future_positions,
                    "future_uncertainties": future_uncertainties,
                }
            )
        return predictions

    def _estimate_prediction_uncertainty(self, step_idx, speed):
        sensor_uncertainty = self.noise_params["pos_std"] + self.noise_params["lidar_std"]
        motion_uncertainty = 0.25 * speed * self.dt * step_idx
        return sensor_uncertainty + motion_uncertainty

    def _compute_uncertainty_aware_safety_reward(self, candidate_pos):
        if not self.prediction_cache:
            return 0.0, float("inf"), self.base_safety_margin

        agent_radius = self.agent_size / 2.0
        total_penalty = 0.0
        min_clearance = float("inf")
        max_margin = self.base_safety_margin

        for pred in self.prediction_cache:
            for horizon_idx, future_pos in enumerate(pred["future_positions"]):
                uncertainty = pred["future_uncertainties"][horizon_idx]
                risk_margin = self.base_safety_margin + uncertainty
                center_dist = np.linalg.norm(candidate_pos - future_pos)
                clearance = center_dist - (agent_radius + pred["radius"])
                shortfall = risk_margin - clearance
                if shortfall > 0.0:
                    total_penalty += (self.prediction_decay ** horizon_idx) * (shortfall ** 2)
                min_clearance = min(min_clearance, clearance)
                max_margin = max(max_margin, risk_margin)

        return self.safety_penalty_gain * total_penalty, min_clearance, max_margin

    def _sample_free_pos(self):
        while True:
            x = self.np_random.uniform(2.0, self.width - 2.0)
            y = self.np_random.uniform(2.0, self.height - 2.0)
            if not self._check_static_collision(np.array([x, y])):
                return np.array([x, y])

    def _generate_complex_map(self):
        grid = np.zeros((100, 100), dtype=np.int8)
        grid[0, :] = 1; grid[-1, :] = 1
        grid[:, 0] = 1; grid[:, -1] = 1
        grid[20:60, 10:15] = 1 
        grid[30:40, 10:25] = 1 
        grid[60:70, 15:25] = 1  
        grid[10:20, 40:50] = 1
        grid[10:15, 60:80] = 1
        # grid[0:15, 65:70] = 1
        # grid[45:50, 0:10] = 1
        grid[25:40, 80:90] = 1 
        # grid[30:35, 85:100] = 1
        grid[10:20, 25:30] = 1
        grid[40:45, 40:70] = 1 
        grid[55:60, 45:75] = 1 
        grid[40:60, 65:70] = 1  
        grid[50:80, 85:90] = 1  
        grid[75:85, 70:85] = 1  
        grid[70:95, 10:40] = 1  
        grid[75:90, 15:35] = 0  
        grid[75:85, 35:40] = 0  
        grid[80:95, 10:15] = 0 

        obstacles = [(25, 35), (70, 55), (90, 20), (50, 25), (65, 80), (90, 60), (80, 45)]
        for (r, c) in obstacles:
            grid[r:r+3, c:c+3] = 1
        return grid

    def _get_info(self):
        return {
            "agent_pos_true": self.agent_pos,
            # "dynamic_obstacles": [o.pos for o in self.dynamic_obstacles]
            "dynamic_obstacles": [o.pos for o in self.dynamic_obstacles],
            "predicted_dynamic_obstacles": [
                {
                    "id": pred["id"],
                    "future_positions": [pos.copy() for pos in pred["future_positions"]],
                    "future_uncertainties": list(pred["future_uncertainties"]),
                }
                for pred in self.prediction_cache
            ],
            "uncertainty_penalty": self.last_safety_metrics["uncertainty_penalty"],
            "risk_min_clearance": self.last_safety_metrics["risk_min_clearance"],
            "risk_margin": self.last_safety_metrics["risk_margin"],
        }
        
    def set_planned_path(self, path):
        self.planned_path = path

    def _draw_dashed_line(self, p1, p2, color=(0, 200, 0), width=2, dash_length=8, gap_length=6):
        total_dist = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
        if total_dist == 0: return
        dx = (p2[0] - p1[0]) / total_dist
        dy = (p2[1] - p1[1]) / total_dist
        current_dist = 0
        while current_dist < total_dist:
            start_x = p1[0] + dx * current_dist
            start_y = p1[1] + dy * current_dist
            end_dist = min(current_dist + dash_length, total_dist)
            end_x = p1[0] + dx * end_dist
            end_y = p1[1] + dy * end_dist
            line = shapes.Line(start_x, start_y, end_x, end_y, width=width, color=color, batch=self.batch)
            self.dashed_lines_cache.append(line)
            current_dist += (dash_length + gap_length)

    def render(self):
        if self.window is None:
            self.window = pyglet.window.Window(width=600, height=600, caption="Improved Uncertain Environment")
            self.batch = pyglet.graphics.Batch()

        self.window.dispatch_events()
        self.window.clear()
        
        scale = 6.0
        shapes.Rectangle(0, 0, 600, 600, color=(255, 255, 255), batch=self.batch).draw()
        
        rows, cols = self.map_matrix.shape
        for r in range(rows):
            for c in range(cols):
                if self.map_matrix[r, c] == 1:
                    shapes.Rectangle(r*scale, c*scale, scale, scale, color=(50, 50, 50), batch=self.batch).draw()
        
        # 动态障碍物 (紫色)
        for obs in self.dynamic_obstacles:
            shapes.Circle(
                x=obs.pos[0]*scale, y=obs.pos[1]*scale, 
                radius=obs.radius*scale, 
                color=(148, 0, 211), batch=self.batch
            ).draw()

        shapes.Circle(self.target_pos[0]*scale, self.target_pos[1]*scale, 2.5*scale, color=(0, 200, 0), batch=self.batch).draw()
        
        if self.planned_path is not None and len(self.planned_path) > 1:
            self.dashed_lines_cache = [] 
            for i in range(len(self.planned_path) - 1):
                p1 = self.planned_path[i] * scale
                p2 = self.planned_path[i+1] * scale
                self._draw_dashed_line(p1, p2, color=(0, 200, 0))

        if len(self.trajectory) > 1:
            for i in range(len(self.trajectory)-1):
                p1 = self.trajectory[i] * scale
                p2 = self.trajectory[i+1] * scale
                shapes.Line(p1[0], p1[1], p2[0], p2[1], width=2, color=(100, 149, 237), batch=self.batch).draw()
        
        cx, cy = self.agent_pos * scale
        half_size = (self.agent_size * scale) / 2
        theta = self.agent_dir
        
        poly_points = []
        for lx, ly in [(half_size, half_size), (-half_size, half_size), (-half_size, -half_size), (half_size, -half_size)]:
            rx = lx * np.cos(theta) - ly * np.sin(theta)
            ry = lx * np.sin(theta) + ly * np.cos(theta)
            poly_points.append((cx + rx, cy + ry))
            
        shapes.Polygon(*poly_points, color=(0, 0, 255), batch=self.batch).draw()
        
        # =========================================================
        # [新增] 绘制车头方向线 (红色)
        # =========================================================
        # 线条长度设为车身大小，使其稍微伸出车体，方便辨认
        heading_len = (self.agent_size * scale) * 1.0
        
        # 计算车头线条终点坐标
        head_x = cx + heading_len * np.cos(theta)
        head_y = cy + heading_len * np.sin(theta)
        
        # 绘制线条 (起点:中心 -> 终点:车头方向)
        shapes.Line(cx, cy, head_x, head_y, width=3, color=(255, 0, 0), batch=self.batch).draw()
        # =========================================================

        self.batch.draw()
        self.window.flip()

    def close(self):
        if self.window:
            self.window.close()

if __name__ == "__main__":
    env = UncertainComplexEnv(render_mode="human")
    obs, info = env.reset()
    print("Environment with Improved Reward Function Loaded.")
    
    for i in range(500):
        # 随机动作测试
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        if terminated:
            print(f"Finished. Final Reward: {reward}")
            break
    env.close()