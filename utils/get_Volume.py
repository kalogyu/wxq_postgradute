import math
import os
import py7zr
import pandas as pd
from pathlib import Path
from typing import Union, List, Dict, Tuple

def calculate_dataset_size(dataset_path: Union[str, Path]) -> float:
    """
    计算数据集的总大小（以GB为单位）。
    支持计算单个文件、目录或压缩文件的大小。

    Args:
        dataset_path (Union[str, Path]): 数据集文件或目录的路径

    Returns:
        float: 数据集大小（GB）
    """
    path = Path(dataset_path)
    total_size = 0

    if not path.exists():
        raise FileNotFoundError(f"路径不存在: {dataset_path}")

    # 如果是压缩文件
    if path.suffix.lower() == '.7z':
        try:
            with py7zr.SevenZipFile(path, mode='r') as z:
                # 获取压缩文件中的文件大小总和
                total_size = sum(info.uncompressed for info in z.list())
        except Exception as e:
            print(f"处理压缩文件时出错: {e}")
            # 如果无法读取压缩文件内容，则使用压缩文件本身的大小
            total_size = path.stat().st_size
    # 如果是目录
    elif path.is_dir():
        for file_path in path.rglob('*'):
            if file_path.is_file():
                total_size += file_path.stat().st_size
    # 如果是单个文件
    else:
        total_size = path.stat().st_size

    # 转换为GB
    return total_size / (1024 * 1024 * 1024)

def calculate_excel_size(file_path: Union[str, Path]) -> Tuple[float, Dict]:
    """
    计算Excel文件的大小并获取其基本信息。

    Args:
        file_path (Union[str, Path]): Excel文件的路径

    Returns:
        Tuple[float, Dict]: (文件大小(GB), 文件信息字典)
    """
    path = Path(file_path)
    
    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")
    
    if path.suffix.lower() not in ['.xlsx', '.xls']:
        raise ValueError(f"不是Excel文件: {file_path}")

    # 获取文件大小（字节）
    file_size_bytes = path.stat().st_size
    
    # 读取Excel文件信息
    try:
        df = pd.read_excel(file_path)
        info = {
            'rows': len(df),
            'columns': len(df.columns),
            'column_names': list(df.columns),
            'memory_usage': df.memory_usage(deep=True).sum(),
            'file_size_bytes': file_size_bytes
        }
    except Exception as e:
        print(f"读取Excel文件时出错: {e}")
        info = {
            'file_size_bytes': file_size_bytes,
            'error': str(e)
        }
    
    # 转换为GB
    size_gb = file_size_bytes / (1024 * 1024 * 1024)
    return size_gb, info

def quantify_volume(dataset_size_gb: float, max_relevant_size_gb: float = 1000.0) -> float:
    """
    计算数据集容量得分。

    得分与数据集大小正相关，但采用对数函数模拟边际递减效应。
    得分将被归一化到0到1之间。

    Args:
        dataset_size_gb (float): 数据集的大小，单位为GB。
        max_relevant_size_gb (float): 认为数据价值达到饱和或最大相关性的参考大小（GB）。
                                      超过此大小的数据量可能价值增长放缓。
                                      这个之后可以根据不同种类的数据集来设置。

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

def get_dataset_volume_score(dataset_path: Union[str, Path], max_relevant_size_gb: float = 1000.0) -> tuple[float, float]:
    """
    计算数据集的容量得分。

    Args:
        dataset_path (Union[str, Path]): 数据集文件或目录的路径
        max_relevant_size_gb (float): 最大相关大小（GB）

    Returns:
        tuple[float, float]: (数据集大小(GB), 容量得分)
    """
    size_gb = calculate_dataset_size(dataset_path)
    score = quantify_volume(size_gb, max_relevant_size_gb)
    return size_gb, score

def get_excel_volume_score(file_path: Union[str, Path], max_relevant_size_gb: float = 1000.0) -> Dict:
    """
    计算Excel文件的容量得分和详细信息。

    Args:
        file_path (Union[str, Path]): Excel文件的路径
        max_relevant_size_gb (float): 最大相关大小（GB）

    Returns:
        Dict: 包含文件大小、容量得分和详细信息的字典
    """
    size_gb, info = calculate_excel_size(file_path)
    score = quantify_volume(size_gb, max_relevant_size_gb)
    
    result = {
        'file_path': str(file_path),
        'size_gb': size_gb,
        'volume_score': score,
        'file_info': info
    }
    
    return result

# 示例用法：
if __name__ == "__main__":
    # 测试单个数据集
    # dataset_path = "数据集/资产负债表.7z"
    # size_gb, score = get_dataset_volume_score(dataset_path)
    # print(f"\n数据集: {dataset_path}")
    # print(f"大小: {size_gb:.2f} GB")
    # print(f"容量得分: {score:.4f}")

    # 测试Excel文件
    excel_path = "数据集/高管及员工薪酬/高管及员工薪酬.xlsx"
    try:
        result = get_excel_volume_score(excel_path)
        print(f"\n数据集: {result['file_path']}")
        print(f"大小: {result['size_gb']:.4f} GB")
        print(f"容量得分: {result['volume_score']:.4f}")
        print("\n文件详细信息:")
        print(f"行数: {result['file_info'].get('rows', 'N/A')}")
        print(f"列数: {result['file_info'].get('columns', 'N/A')}")
        print(f"列名: {result['file_info'].get('column_names', 'N/A')}")
        print(f"内存使用: {result['file_info'].get('memory_usage', 'N/A')} bytes")
    except Exception as e:
        print(f"处理文件时出错: {e}")

    # # 测试不同大小的数据集
    # print("\n不同大小的数据集得分示例：")
    # test_sizes = [0.001, 0.01, 0.1, 1, 10, 100]  # 更符合Excel文件大小的范围
    # for size in test_sizes:
    #     score = quantify_volume(size)
    #     print(f"{size:.3f} GB 数据集的容量得分: {score:.4f}")