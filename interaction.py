# def unscale_action(scaled_action):
#     """
#     将网络输出的 [-1, 1] 映射回环境的真实物理量
#     Network Output: [u_v, u_w] in [-1, 1]
#     Target: v in [-1, 2], w in [-1, 1]
#     """
#     v_scaled, w_scaled = scaled_action[..., 0], scaled_action[..., 1]
    
#     # 映射 v: [-1, 1] -> [-1, 2]
#     # 公式: out = (in + 1) / 2 * (max - min) + min
#     v_real = (v_scaled + 1.0) / 2.0 * (2.0 - (-1.0)) + (-1.0)
    
#     # 映射 w: [-1, 1] -> [-1, 1] (无需变化)
#     w_real = w_scaled
    
#     return torch.stack([v_real, w_real], dim=-1)