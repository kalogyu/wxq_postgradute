import numpy as np
import matplotlib.pyplot as plt

def s_volume_function(V, V_optimal, k):
    """
    计算数据集容量得分函数 S_Volume(V)。
    Args:
        V (float or np.array): 数据量。
        V_optimal (float): 最佳数据量，曲线的拐点。
        k (float): 陡峭度参数。
    Returns:
        float or np.array: 数据量得分 (0-1)。
    """
    return 1 / (1 + np.exp(-k * (V - V_optimal)))

# 示例参数
V_optimal_example = 5000  # 假设最佳数据量为 5000 条记录
k_example = 0.002         # 陡峭度参数，控制曲线的上升速度

# 生成数据量范围 (横坐标)
# 从 0 到 15000 条记录，用于展示曲线的完整形态
data_volumes = np.linspace(0, 15000, 500)

# 计算对应的得分 (纵坐标)
scores = s_volume_function(data_volumes, V_optimal_example, k_example)

# 绘制图形
plt.figure(figsize=(10, 6))
plt.plot(data_volumes, scores, label=f'S_Volume(V) with V_optimal={V_optimal_example}, k={k_example}')
plt.axvline(x=V_optimal_example, color='r', linestyle='--', label=f'V_optimal = {V_optimal_example} (Score = 0.5)')
plt.axhline(y=0.5, color='gray', linestyle=':', label='Score = 0.5')

plt.title('数据集容量得分函数：数据量与得分的关系')
plt.xlabel('数据量 (例如：记录数量)')
plt.ylabel('数据量得分 (0-1)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.ylim(0, 1) # 确保纵坐标范围在0到1之间
plt.show()