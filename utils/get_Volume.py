import math

def quantify_volume(dataset_size_gb: float, max_relevant_size_gb: float = 1000.0) -> float:
    """
    计算数据集容量得分。

    得分与数据集大小正相关，但采用对数函数模拟边际递减效应。
    得分将被归一化到0到1之间。

    Args:
        dataset_size_gb (float): 数据集的大小，单位为GB。
        max_relevant_size_gb (float): 认为数据价值达到饱和或最大相关性的参考大小（GB）。
                                      超过此大小的数据量可能价值增长放缓。

    Returns:
        float: 归一化后的数据容量得分 (0到1之间)。
    """
    if dataset_size_gb <= 0:
        return 0.0

    # 使用对数函数来模拟边际递减效应
    # log(x+1) 确保当 dataset_size_gb 接近0时，结果不会是负无穷
    log_scaled_size = math.log10(dataset_size_gb + 1)

    # 归一化因子，基于 max_relevant_size_gb
    # 确保当 dataset_size_gb 达到 max_relevant_size_gb 时，得分接近1
    normalization_factor = math.log10(max_relevant_size_gb + 1)

    if normalization_factor == 0: # 避免除以零
        return 0.0

    score = log_scaled_size / normalization_factor

    # 确保得分在0到1之间
    return max(0.0, min(score, 1.0))

# 示例用法：
print(f"1 GB 数据集的容量得分: {quantify_volume(1):.4f}")
print(f"10 GB 数据集的容量得分: {quantify_volume(10):.4f}")
print(f"100 GB 数据集的容量得分: {quantify_volume(100):.4f}")
print(f"500 GB 数据集的容量得分: {quantify_volume(500):.4f}")
print(f"1000 GB (1 TB) 数据集的容量得分: {quantify_volume(1000):.4f}")
print(f"2000 GB (2 TB) 数据集的容量得分: {quantify_volume(2000):.4f}")
print(f"0 GB 数据集的容量得分: {quantify_volume(0):.4f}")