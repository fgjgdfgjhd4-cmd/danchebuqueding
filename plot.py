import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. 配置参数 (请根据您的实际情况修改)
# ==========================================
# CSV 文件路径
FILE_AC = "AC_GDPO_Curriculum_20260210_173117.csv"  # 您的 AC-GDPO 数据文件名
FILE_PPO = "PPO_Curriculum_20260210_204806.csv"     # 您的 PPO 数据文件名

# 课程学习的阶段分界线 (Episode)
# 根据您的训练代码设置：Stage 1 结束于 200, Stage 2 结束于 600
STAGE_BOUNDARIES = [100, 400, 900] 

# 阶段名称
STAGE_LABELS = ["Easy", "Medium", "Hard", "Random Long (>100m)"]

# 数据平滑系数 (0~1, 越大越平滑)
SMOOTH_FACTOR = 0.9

# ==========================================
# 2. 辅助函数: 数据平滑 (类似 TensorBoard)
# ==========================================
def smooth(scalars, weight):
    last = scalars[0]
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return np.array(smoothed)

# ==========================================
# 3. 读取与处理数据
# ==========================================
try:
    # 读取 CSV (TensorBoard 导出的 CSV 通常有 'Step' 和 'Value' 列)
    df_ac = pd.read_csv(FILE_AC)
    df_ppo = pd.read_csv(FILE_PPO)
    
    # 提取数据
    steps_ac = df_ac['Step']
    rew_ac = df_ac['Value']
    
    steps_ppo = df_ppo['Step']
    rew_ppo = df_ppo['Value']

    # 平滑处理 (原始数据通常波动很大，平滑后更好看)
    rew_ac_smooth = smooth(rew_ac, SMOOTH_FACTOR)
    rew_ppo_smooth = smooth(rew_ppo, SMOOTH_FACTOR)

except FileNotFoundError as e:
    print(f"错误: 找不到文件 {e.filename}。请确保 CSV 文件已下载并放在同一目录下。")
    exit()

# ==========================================
# 4. 绘图核心逻辑
# ==========================================
plt.figure(figsize=(12, 6), dpi=150) # 设置画布大小和清晰度

# --- A. 绘制曲线 ---
# AC-GDPO (红色系)
plt.plot(steps_ac, rew_ac, color='salmon', alpha=0.2) # 原始数据(半透明)
plt.plot(steps_ac, rew_ac_smooth, color='red', label='AC-GDPO', linewidth=2) # 平滑数据

# PPO (蓝色系)
plt.plot(steps_ppo, rew_ppo, color='lightblue', alpha=0.2)
plt.plot(steps_ppo, rew_ppo_smooth, color='blue', label='PPO', linewidth=2)

# --- B. 绘制阶段分割线与背景 ---
# 获取Y轴范围，用于放置文字
y_min, y_max = plt.ylim()
if y_min > -50: y_min = -50 # 稍微调整下限以免文字太靠底
if y_max < 300: y_max = 300

# 添加分割线
for x in STAGE_BOUNDARIES:
    plt.axvline(x=x, color='gray', linestyle='--', linewidth=1.5, alpha=0.8)

# 添加阶段背景色和文字
# 获取 X 轴的总长度 (取两者中较长的那个)
max_step = max(steps_ac.iloc[-1], steps_ppo.iloc[-1])
boundaries = [0] + STAGE_BOUNDARIES + [max_step]

colors = ['#f9f9f9', '#eef9e3', '#e3f2fd'] # 灰白、浅绿、浅蓝 (对应三个阶段)

for i in range(len(boundaries) - 1):
    start = boundaries[i]
    end = boundaries[i+1]
    
    # 1. 填充背景色
    if i < len(colors):
        plt.axvspan(start, end, color=colors[i], alpha=0.3)
    
    # 2. 添加阶段文字标签
    if i < len(STAGE_LABELS):
        mid_point = (start + end) / 2
        plt.text(mid_point, y_max * 0.95, STAGE_LABELS[i], 
                 ha='center', va='top', fontsize=12, fontweight='bold', color='dimgray')

# --- C. 图表装饰 ---
plt.title("Training Performance Comparison: AC-GDPO vs PPO", fontsize=16, pad=20)
plt.xlabel("Episodes", fontsize=12)
plt.ylabel("Average Reward", fontsize=12)
plt.legend(loc='lower right', frameon=True, shadow=True, fontsize=12)
plt.grid(True, linestyle=':', alpha=0.6)

# 限制 X 轴范围
plt.xlim(0, max_step)

# 保存与显示
plt.tight_layout()
plt.savefig("comparison_result.png") # 保存为图片
plt.show() # 显示窗口