import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.special import erf
import time

# 引入 Pyglet 用于在环境窗口中绘图
from pyglet import shapes

class BeliefNode:
    """
    信念节点，对应论文中的 b_t = (x_t, Sigma_t)
    """
    def __init__(self, x, y, cov=None, cost=0.0, parent=None):
        self.x = x
        self.y = y
        self.pos = np.array([x, y])
        # 协方差矩阵 (2x2)，如果未指定则初始化为零矩阵
        self.cov = cov if cov is not None else np.zeros((2, 2))
        self.cost = cost # 到达该节点的代价 (Cost function, Eq. 15)
        self.risk = 0.0  # 该节点的碰撞风险
        self.parent = parent
        
    def __repr__(self):
        return f"Node({self.x:.1f}, {self.y:.1f}, Cost={self.cost:.1f})"

class RA_RRTV_Planner:
    """
    RA-RRTV* 算法复现
    Risk-Averse RRT* with Local Vine Expansion
    """
    def __init__(self, env, max_iter=1000, step_size=3.0, search_radius=10.0):
        self.env = env
        self.map_matrix = env.map_matrix
        self.width = env.width
        self.height = env.height
        
        # --- 算法参数 (参考论文 Table I) ---
        self.max_iter = max_iter
        self.step_size = step_size     # r_step
        self.search_radius = search_radius # r_min
        
        # 风险权重参数 (Eq. 15)
        self.alpha = 5.0 # 累积风险权重
        self.beta = 5.0  # 最大风险权重
        
        # 约束阈值
        self.delta_s = 0.8 # 安全概率阈值 (对应论文 1 - delta_collision)
        self.delta_p = 5.0 # 协方差迹阈值 (不确定性上限)
        
        # Vine 相关参数
        self.fail_threshold = 10 # 触发 Vine 的失败次数阈值 (epsilon_n)
        self.failed_samples = [] # X_fail
        
        # 构建障碍物 KDTree 用于快速查询最近障碍物距离
        self.obs_tree = self._build_obstacle_tree()
        
        # 树结构
        self.start_node = None
        self.goal_node = None
        self.node_list = []
        
        # 系统噪声矩阵 (对应环境中的 noise_params)
        # 假设简单的线性传播 Q
        self.Q = np.eye(2) * (env.noise_params["pos_std"] ** 2)
        # 测量噪声 R (假设在某些区域可以校准，这里简化为全局存在)
        self.R = np.eye(2) * 0.01 

    def _build_obstacle_tree(self):
        """构建 KDTree 加速距离计算"""
        y_idxs, x_idxs = np.where(self.map_matrix == 1)
        # 注意：map_matrix索引是 [x, y], 但坐标系可能是对应的
        # 环境中 map_matrix[ix, iy]，ix是x坐标。
        coords = np.column_stack((x_idxs, y_idxs))
        if len(coords) == 0:
            return None
        return cKDTree(coords)

    def plan(self, start_pos, goal_pos):
        """主规划函数 (Algorithm 1)"""
        print(f"[RA-RRTV*] Start Planning from {start_pos} to {goal_pos}")
        
        self.start_node = BeliefNode(start_pos[0], start_pos[1], cov=np.eye(2)*0.1)
        self.goal_pos = np.array(goal_pos)
        self.node_list = [self.start_node]
        self.failed_samples = []
        
        for i in range(self.max_iter):
            # 1. 采样 (SampleFree)
            rnd_pos = self._sample_free()
            
            # 2. 最近邻 (Nearest)
            nearest_ind = self._get_nearest_node_index(self.node_list, rnd_pos)
            nearest_node = self.node_list[nearest_ind]
            
            # 3. 扩展 (Steer)
            new_node = self._steer(nearest_node, rnd_pos, self.step_size)
            
            # 4. 风险与碰撞检测 (RiskFeas)
            if self._check_collision(new_node) and self._check_risk_constraints(new_node):
                # 寻找附近节点 (Near)
                near_inds = self._find_near_nodes(new_node)
                
                # 选择最佳父节点 (Choose Parent)
                new_node = self._choose_parent(new_node, near_inds)
                
                if new_node:
                    self.node_list.append(new_node)
                    # 重连 (Rewire)
                    self._rewire(new_node, near_inds)
                    
                    # 检查是否到达终点
                    if self._calc_dist(new_node.pos, self.goal_pos) <= self.step_size + 1.0:
                        # 尝试直接连接终点
                        final_node = self._steer(new_node, self.goal_pos, self.step_size)
                        if self._check_collision(final_node):
                             print(f"[RA-RRTV*] Goal Reached at iter {i}!")
                             self.goal_node = final_node
                             return self._generate_path(self.goal_node)
            else:
                # 记录失败样本
                self.failed_samples.append(new_node.pos)
            
            # 5. Local Vine Expansion (算法核心创新)
            # 当失败样本过多时，说明遇到了"狭窄通道" (Probabilistic Narrow Passage)
            if len(self.failed_samples) > self.fail_threshold:
                self._trigger_local_vine()
                self.failed_samples = [] # 重置

            if i % 100 == 0:
                print(f"Iter: {i}, Nodes: {len(self.node_list)}")

        return None # 未找到路径

    def _sample_free(self):
        """随机采样，有一定概率直接偏向终点"""
        if np.random.rand() > 0.05:
            return np.array([
                np.random.uniform(0, self.width),
                np.random.uniform(0, self.height)
            ])
        else:
            return self.goal_pos

    def _steer(self, from_node, to_pos, extend_length=float("inf")):
        """
        延伸节点并传播不确定性 (Steer + Propagate Belief)
        对应论文 Eq. 5-9 (卡尔曼滤波预测步骤)
        """
        new_node = BeliefNode(from_node.x, from_node.y)
        d, theta = self._calc_dist_and_angle(new_node, to_pos)

        new_node.dist = min(extend_length, d)
        new_node.x += new_node.dist * np.cos(theta)
        new_node.y += new_node.dist * np.sin(theta)
        new_node.pos = np.array([new_node.x, new_node.y])

        new_node.parent = from_node

        # --- 传播不确定性 (简化 KF Predict) ---
        # Sigma_t = A * Sigma_{t-1} * A^T + Q
        # 假设 A=I (线性运动), Q 为过程噪声
        new_node.cov = from_node.cov + self.Q 
        
        # 如果环境有观测 (这里简化模拟：每隔一定距离不确定性收敛)
        # 对应 Eq. 8 (KF Update)
        # 简单模拟：如果靠近某些"特征区"（这里简化为每步都做一点修正，模拟 continuous sensing）
        K = 0.1 # 简化的卡尔曼增益
        new_node.cov = (np.eye(2) - K * np.eye(2)) @ new_node.cov

        # 计算风险和代价
        new_node.risk = self._calc_collision_prob(new_node)
        
        # Cost Function (Eq. 15): 
        # C = Dist + alpha * Sum(Risk) + beta * Max(Risk)
        step_dist = np.linalg.norm(new_node.pos - from_node.pos)
        
        # 递归累加父节点的风险
        parent_accum_risk = (from_node.cost - self._calc_dist_cost_only(from_node)) / self.alpha if self.alpha > 0 else 0
        current_accum_risk = parent_accum_risk + new_node.risk
        
        # 简化处理：这里主要计算 Cost 用于 RRT* 比较
        # 实际代价 = 距离代价 + alpha * 累积风险
        # (beta项在局部优化中体现，为简化计算暂略)
        new_node.cost = from_node.cost + step_dist + self.alpha * new_node.risk

        return new_node
    
    def _calc_dist_cost_only(self, node):
        """辅助函数：回溯计算纯距离代价"""
        d = 0
        curr = node
        while curr.parent:
            d += np.linalg.norm(curr.pos - curr.parent.pos)
            curr = curr.parent
        return d

    def _calc_collision_prob(self, node):
        """
        计算碰撞概率 (Eq. 12)
        P = (1 - erf(dist / (sigma * sqrt(2)))) / 2
        """
        if self.obs_tree is None:
            return 0.0
            
        # 找到最近障碍物的距离
        dist, _ = self.obs_tree.query(node.pos)
        
        # 获取在该方向上的不确定性 (投影到最近障碍物方向)
        # 简化：使用协方差矩阵的最大特征值作为保守估计 sigma
        eigenvalues = np.linalg.eigvals(node.cov)
        sigma = np.sqrt(np.max(eigenvalues))
        
        if sigma < 1e-6: return 0.0
        
        # 考虑到 agent radius (2.0)
        margin = dist - self.env.agent_size / 2.0
        
        # 如果已经在障碍物内，概率为 1.0
        if margin <= 0: return 1.0
        
        # 使用误差函数计算概率
        prob = 0.5 * (1 - erf(margin / (sigma * np.sqrt(2))))
        return prob

    def _check_risk_constraints(self, node):
        """检查机会约束 (Eq. 3 & 4)"""
        # 1. 碰撞概率约束
        if (1.0 - node.risk) < self.delta_s:
            return False
            
        # 2. 不确定性发散约束 (Trace of Covariance)
        if np.trace(node.cov) > self.delta_p:
            return False
            
        return True

    def _check_collision(self, node):
        """几何碰撞检测 (硬约束)"""
        # 检查是否出界
        if node.x < 0 or node.x >= self.width or \
           node.y < 0 or node.y >= self.height:
            return False
            
        # 检查是否撞墙 (静态)
        ix, iy = int(node.x), int(node.y)
        if self.map_matrix[ix, iy] == 1:
            return False
            
        # 检查路径连线碰撞 (简单插值)
        if node.parent:
            steps = 5
            for i in range(steps):
                u = i / steps
                x = node.parent.x * (1-u) + node.x * u
                y = node.parent.y * (1-u) + node.y * u
                if self.map_matrix[int(x), int(y)] == 1:
                    return False
        return True

    def _trigger_local_vine(self):
        """
        Local Belief Vine (Algorithm 2)
        当检测到狭窄通道时（采样频繁失败），触发此策略。
        """
        if not self.failed_samples:
            return

        # 1. 聚类失败点，找到潜在的"狭窄通道中心" (x_potn)
        # 简化：直接取所有失败点的均值中心
        failures = np.array(self.failed_samples)
        x_potn = np.mean(failures, axis=0)
        
        # 2. 在树中找到离 x_potn 最近的节点作为 Vine Root
        nearest_ind = self._get_nearest_node_index(self.node_list, x_potn)
        vine_root = self.node_list[nearest_ind]
        
        # 3. 定向生长 (Vine Growth)
        # 论文中使用 Von Mises-Fisher 分布进行定向采样
        # 这里模拟这一行为：在 root 指向 x_potn 的锥形区域内密集采样
        vec = x_potn - vine_root.pos
        target_angle = math.atan2(vec[1], vec[0])
        
        # 尝试生成 N 个节点，形成 Vine
        num_vine_nodes = 10
        curr_node = vine_root
        
        for _ in range(num_vine_nodes):
            # 在目标方向附近采样 (模拟贝叶斯序列采样的结果)
            angle_noise = np.random.normal(0, 0.2) # 小方差
            sample_angle = target_angle + angle_noise
            
            # 步长稍微小一点，精细探索
            vine_step = self.step_size * 0.8
            
            next_x = curr_node.x + vine_step * np.cos(sample_angle)
            next_y = curr_node.y + vine_step * np.sin(sample_angle)
            
            # 尝试生成节点
            vine_node = self._steer(curr_node, np.array([next_x, next_y]), vine_step)
            
            if self._check_collision(vine_node) and self._check_risk_constraints(vine_node):
                self.node_list.append(vine_node)
                curr_node = vine_node # 继续生长
                
                # 稍微修正方向指向终点，避免跑偏
                vec_to_goal = self.goal_pos - curr_node.pos
                target_angle = math.atan2(vec_to_goal[1], vec_to_goal[0])
            else:
                break # 撞墙则停止生长

    def _get_nearest_node_index(self, node_list, rnd_pos):
        dlist = [(node.x - rnd_pos[0])**2 + (node.y - rnd_pos[1])**2 for node in node_list]
        minind = dlist.index(min(dlist))
        return minind

    def _find_near_nodes(self, new_node):
        nnode = len(self.node_list) + 1
        r = self.search_radius * np.sqrt((np.log(nnode) / nnode))
        r = min(r, self.search_radius) # 限制最大搜索半径
        dist_list = [(node.x - new_node.x)**2 + (node.y - new_node.y)**2 for node in self.node_list]
        near_inds = [dist_list.index(i) for i in dist_list if i <= r**2]
        return near_inds

    def _choose_parent(self, new_node, near_inds):
        if not near_inds:
            return new_node
            
        costs = []
        for i in near_inds:
            near_node = self.node_list[i]
            t_node = self._steer(near_node, new_node.pos) # 重新计算连接代价
            if t_node and self._check_collision(t_node) and self._check_risk_constraints(t_node):
                costs.append(t_node.cost)
            else:
                costs.append(float("inf"))
                
        min_cost = min(costs)
        if min_cost == float("inf"):
            return None
            
        min_ind = near_inds[costs.index(min_cost)]
        new_node = self._steer(self.node_list[min_ind], new_node.pos)
        return new_node

    def _rewire(self, new_node, near_inds):
        for i in near_inds:
            near_node = self.node_list[i]
            edge_node = self._steer(new_node, near_node.pos)
            
            if edge_node:
                # 如果通过新节点到达 near_node 的代价更低
                if edge_node.cost < near_node.cost:
                    if self._check_collision(edge_node) and self._check_risk_constraints(edge_node):
                        near_node.parent = new_node
                        near_node.cost = edge_node.cost
                        near_node.cov = edge_node.cov # 更新不确定性流

    def _generate_path(self, goal_node):
        path = []
        node = goal_node
        while node.parent is not None:
            path.append([node.x, node.y])
            node = node.parent
        path.append([node.x, node.y])
        return np.array(path[::-1]) # 反转路径

    def _calc_dist(self, pos1, pos2):
        return np.linalg.norm(pos1 - pos2)
    
    def _calc_dist_and_angle(self, from_node, to_pos):
        dx = to_pos[0] - from_node.x
        dy = to_pos[1] - from_node.y
        d = math.hypot(dx, dy)
        theta = math.atan2(dy, dx)
        return d, theta

    # --- 可视化相关 ---
    def draw_tree(self, batch):
        """将 RRT 树绘制到 Pyglet batch 中"""
        self.vertex_shapes = []
        self.edge_shapes = []
        
        for node in self.node_list:
            if node.parent:
                # 画边
                line = shapes.Line(node.x * 6.0, node.y * 6.0, 
                                   node.parent.x * 6.0, node.parent.y * 6.0, 
                                   width=1, color=(200, 200, 200), batch=batch)
                self.edge_shapes.append(line)
            # 画节点 (可选，节点太多会卡)
            # circle = shapes.Circle(node.x * 6.0, node.y * 6.0, radius=2, color=(0, 255, 0), batch=batch)
            # self.vertex_shapes.append(circle)
            
# ==========================================
# 主运行逻辑
# ==========================================
if __name__ == "__main__":
    from uncertain_env import UncertainComplexEnv
    
    # 1. 初始化环境
    # 设置一个稍微难一点的起点和终点，跨越障碍区
    start = [10.0, 10.0]
    target = [90.0, 90.0]
    
    env = UncertainComplexEnv(render_mode="human", start_pos=start, target_pos=target)
    obs, info = env.reset()
    
    # 2. 初始化 RA-RRTV* 规划器
    planner = RA_RRTV_Planner(env, max_iter=2000, step_size=2.0)
    
    print("Initializing Planner...")
    print("Generating Path... This may take a few seconds (Python RRT is slow).")
    
    # 3. 执行规划
    start_time = time.time()
    path = planner.plan(start, target)
    end_time = time.time()
    
    if path is not None:
        print(f"Path found! Length: {len(path)} nodes. Time: {end_time - start_time:.2f}s")
        env.set_planned_path(path)
    else:
        print("Failed to find path within max iterations.")

    # 4. 运行可视化循环
    # 为了看到生成的树，我们利用环境的 render 循环，并注入我们的绘制代码
    
    # 我们需要一种方式将树传递给环境渲染，或者在环境渲染后覆盖渲染
    # 这里通过简单的 Hack：利用环境的 window 和 batch
    
    print("Visualizing...")
    
    for i in range(1000): # 保持窗口打开一段时间
        env.render()
        
        # 实时绘制树结构 (覆盖在地图上)
        # 注意：每一帧都重建 shape 对象非常耗费资源，这里仅作演示
        # 实际应用中应只创建一次
        if planner.node_list:
             planner.draw_tree(env.batch)
             env.batch.draw() # 再次绘制 Batch 以显示树
             env.window.flip()
        
        # 简单的路径跟踪控制 (纯演示，非算法核心)
        if path is not None and i < 500:
            # 简单 P 控制器追随路径
            curr_pos = env.agent_pos
            # 找路径上最近的点的前方一点
            dists = np.linalg.norm(path - curr_pos, axis=1)
            target_idx = np.argmin(dists) + 1
            if target_idx >= len(path): target_idx = len(path) - 1
            
            waypoint = path[target_idx]
            angle = math.atan2(waypoint[1] - curr_pos[1], waypoint[0] - curr_pos[0])
            
            # 计算动作
            current_angle = env.agent_dir
            diff = angle - current_angle
            diff = (diff + np.pi) % (2 * np.pi) - np.pi
            
            v = 1.0 if np.linalg.norm(waypoint - curr_pos) > 2.0 else 0.0
            w = diff * 2.0
            
            env.step(np.array([v, w]))
            
        elif path is None:
             pass