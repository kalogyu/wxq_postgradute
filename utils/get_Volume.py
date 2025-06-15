import math
import os
import py7zr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Union, List, Dict, Tuple, Optional
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VolumeScorer:
    """
    数据量评分器类，用于计算数据集的数据量得分
    使用改进的Sigmoid函数来评估数据量对数据集容量的贡献
    """
    
    def __init__(self, 
                 v_optimal: float = 1000.0,  # 最佳数据量阈值
                 k: float = 0.001,          # 曲线陡峭度参数
                 min_volume: float = 100.0,  # 最小有效数据量
                 max_volume: float = 1000000.0):  # 最大有效数据量
        """
        初始化数据量评分器
        
        Args:
            v_optimal: 最佳数据量阈值，得分曲线的拐点
            k: 曲线陡峭度参数，控制得分增长速度
            min_volume: 最小有效数据量
            max_volume: 最大有效数据量
        """
        self.v_optimal = v_optimal
        self.k = k
        self.min_volume = min_volume
        self.max_volume = max_volume
        
    def _normalize_volume(self, volume: float) -> float:
        """
        将数据量标准化到有效范围内
        
        Args:
            volume: 原始数据量
            
        Returns:
            标准化后的数据量
        """
        return np.clip(volume, self.min_volume, self.max_volume)
    
    def _sigmoid_score(self, volume: float) -> float:
        """
        使用改进的Sigmoid函数计算数据量得分
        
        Args:
            volume: 标准化后的数据量
            
        Returns:
            数据量得分 (0-1之间)
        """
        normalized_volume = self._normalize_volume(volume)
        score = 1 / (1 + np.exp(-self.k * (normalized_volume - self.v_optimal)))
        return float(score)
    
    def calculate_volume_score(self, 
                             data: Union[pd.DataFrame, str],
                             volume_type: str = 'rows') -> Dict[str, float]:
        """
        计算数据集的数据量得分
        
        Args:
            data: 数据集（DataFrame或文件路径）
            volume_type: 数据量类型 ('rows' 或 'size')
            
        Returns:
            包含数据量得分的字典
        """
        try:
            # 获取数据量
            if isinstance(data, str):
                if not os.path.exists(data):
                    raise FileNotFoundError(f"文件不存在: {data}")
                
                if volume_type == 'rows':
                    volume = len(pd.read_csv(data))
                else:  # size
                    volume = os.path.getsize(data) / (1024 * 1024)  # 转换为MB
            else:
                if volume_type == 'rows':
                    volume = len(data)
                else:  # size
                    # 估算DataFrame的内存大小（MB）
                    volume = data.memory_usage(deep=True).sum() / (1024 * 1024)
            
            # 计算得分
            score = self._sigmoid_score(volume)
            
            return {
                'volume': volume,
                'score': score,
                'normalized_volume': self._normalize_volume(volume)
            }
            
        except Exception as e:
            logger.error(f"计算数据量得分时发生错误: {str(e)}")
            raise
    
    def get_volume_metrics(self, 
                          data: Union[pd.DataFrame, str],
                          volume_type: str = 'rows') -> Dict[str, float]:
        """
        获取详细的数据量指标
        
        Args:
            data: 数据集（DataFrame或文件路径）
            volume_type: 数据量类型 ('rows' 或 'size')
            
        Returns:
            包含详细数据量指标的字典
        """
        try:
            # 计算基本得分
            basic_metrics = self.calculate_volume_score(data, volume_type)
            
            # 计算额外指标
            volume = basic_metrics['volume']
            normalized_volume = basic_metrics['normalized_volume']
            score = basic_metrics['score']
            
            # 计算与最佳值的差距
            gap_to_optimal = abs(volume - self.v_optimal)
            
            # 计算数据量效率（当前得分与数据量的比值）
            efficiency = score / normalized_volume if normalized_volume > 0 else 0
            
            return {
                'raw_volume': volume,
                'normalized_volume': normalized_volume,
                'score': score,
                'gap_to_optimal': gap_to_optimal,
                'efficiency': efficiency,
                'is_optimal': volume >= self.v_optimal,
                'is_minimal': volume >= self.min_volume
            }
            
        except Exception as e:
            logger.error(f"获取数据量指标时发生错误: {str(e)}")
            raise

def get_volume_score(data: Union[pd.DataFrame, str],
                    v_optimal: float = 1000.0,
                    k: float = 0.001,
                    volume_type: str = 'rows') -> Dict[str, float]:
    """
    便捷函数：计算数据集的数据量得分
    
    Args:
        data: 数据集（DataFrame或文件路径）
        v_optimal: 最佳数据量阈值
        k: 曲线陡峭度参数
        volume_type: 数据量类型 ('rows' 或 'size')
        
    Returns:
        包含数据量得分的字典
    """
    scorer = VolumeScorer(v_optimal=v_optimal, k=k)
    return scorer.calculate_volume_score(data, volume_type)

def get_volume_metrics(data: Union[pd.DataFrame, str],
                      v_optimal: float = 1000.0,
                      k: float = 0.001,
                      volume_type: str = 'rows') -> Dict[str, float]:
    """
    便捷函数：获取详细的数据量指标
    
    Args:
        data: 数据集（DataFrame或文件路径）
        v_optimal: 最佳数据量阈值
        k: 曲线陡峭度参数
        volume_type: 数据量类型 ('rows' 或 'size')
        
    Returns:
        包含详细数据量指标的字典
    """
    scorer = VolumeScorer(v_optimal=v_optimal, k=k)
    return scorer.get_volume_metrics(data, volume_type)

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

def quantify_volume(dataset_size_gb: float, max_relevant_size_gb: float = 0.17) -> float:
    """
    计算数据集容量得分。

    使用改进的对数函数计算得分，更好地反映数据价值。
    考虑到当前最大文件约为170MB，将max_relevant_size_gb设置为0.17GB。
    使用对数函数可以更好地反映数据价值的边际递减效应。

    Args:
        dataset_size_gb (float): 数据集的大小，单位为GB。
        max_relevant_size_gb (float): 最大相关大小（GB），默认设为0.17GB（170MB）。

    Returns:
        float: 数据容量得分 (0到100之间)。
    """
    if dataset_size_gb <= 0:
        return 0.0

    # 使用改进的对数函数
    # 1. 将输入值缩放到更合适的范围
    # 2. 使用log10函数计算基础得分
    # 3. 调整系数使最大文件得到接近100分
    scaled_size = dataset_size_gb * 1000  # 转换为MB
    base_score = math.log10(scaled_size + 1)  # 加1避免log(0)
    
    # 计算最大可能得分（当文件大小为max_relevant_size_gb时）
    max_scaled_size = max_relevant_size_gb * 1000
    max_base_score = math.log10(max_scaled_size + 1)
    
    # 归一化并缩放到0-100，调整系数使最大文件得到接近100分
    normalized_score = (base_score / max_base_score) * 120  # 增加系数使最大文件得到更高分
    
    # 添加一个缩放因子，使小文件也能得到合理的分数
    min_score = 15  # 提高最小得分
    score = min_score + (normalized_score - min_score) * (1 - math.exp(-scaled_size/10))  # 调整衰减速率
    
    # 确保得分在0到100之间
    return max(0.0, min(score, 100.0))

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

def get_excel_volume_score(file_path: Union[str, Path], max_relevant_size_gb: float = 0.17) -> Dict:
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
    
    # 转换大小为MB显示
    size_mb = size_gb * 1024
    
    result = {
        'file_path': str(file_path),
        'size_mb': size_mb,  # 使用MB作为显示单位
        'volume_score': score,
        'file_info': info
    }
    
    return result

def plot_volume_analysis(results: List[Dict]):
    """
    绘制文件大小和得分的可视化图表。
    
    Args:
        results: 包含文件分析结果的列表
    """
    # 创建图表
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 准备数据
    sizes = [r['size_mb'] for r in results]
    scores = [r['volume_score'] for r in results]
    names = [r['file_name'] for r in results]
    
    # 1. 散点图：文件大小 vs 得分
    scatter = ax1.scatter(sizes, scores, alpha=0.6)
    ax1.set_xlabel('File Size (MB)')
    ax1.set_ylabel('Volume Score')
    ax1.set_title('File Size vs Volume Score')
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # 添加趋势线
    z = np.polyfit(sizes, scores, 1)
    p = np.poly1d(z)
    ax1.plot(sizes, p(sizes), "r--", alpha=0.8)
    
    # 2. 得分分布直方图
    ax2.hist(scores, bins=10, edgecolor='black')
    ax2.set_xlabel('Volume Score')
    ax2.set_ylabel('Number of Files')
    ax2.set_title('Distribution of Volume Scores')
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图表
    plt.savefig('volume_analysis.png')
    print("\n图表已保存为 'volume_analysis.png'")
    
    # 显示图表
    plt.show()

def plot_score_curve():
    """
    绘制文件大小从0到170MB时的得分变化曲线。
    """
    # 创建文件大小范围（0到170MB，转换为GB）
    sizes_mb = np.linspace(0, 170, 1000)  # 1000个点使曲线更平滑
    sizes_gb = sizes_mb / 1024  # 转换为GB
    
    # 计算每个大小对应的得分
    scores = [quantify_volume(size_gb) for size_gb in sizes_gb]
    
    # 创建图表
    plt.figure(figsize=(10, 6))
    
    # 绘制曲线
    plt.plot(sizes_mb, scores, 'b-', linewidth=2)
    
    # 添加一些关键点的标记
    key_sizes = [0, 10, 50, 100, 170]  # MB
    key_scores = [quantify_volume(size/1024) for size in key_sizes]
    plt.scatter(key_sizes, key_scores, color='red', s=50, zorder=5)
    
    # 为关键点添加标注
    for size, score in zip(key_sizes, key_scores):
        plt.annotate(f'({size}MB, {score:.1f}分)',
                    xy=(size, score),
                    xytext=(10, 10),
                    textcoords='offset points',
                    ha='left',
                    va='bottom',
                    bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5))
    
    # 设置图表属性
    plt.title('文件大小与得分关系曲线', fontsize=12)
    plt.xlabel('文件大小 (MB)', fontsize=10)
    plt.ylabel('Volume Score', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 设置坐标轴范围
    plt.xlim(-5, 175)  # 留出一些边距
    plt.ylim(-5, 105)  # 留出一些边距
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图表
    plt.savefig('score_curve.png')
    print("\n得分曲线图已保存为 'score_curve.png'")
    
    # 显示图表
    plt.show()

# 示例用法：
if __name__ == "__main__":
    # 绘制得分曲线
    plot_score_curve()
    
    # 询问是否要分析文件
    choice = input("\n是否要分析实际文件？(y/n): ").strip().lower()
    
    if choice == 'y':
        # 测试目录下的所有Excel文件
        excel_dir = r"e:\wxq_postgradute\数据集\extracted_excel_files"
        try:
            # 获取目录下所有Excel文件
            excel_files = list(Path(excel_dir).glob('*.xlsx'))
            
            if not excel_files:
                print(f"No Excel files found in {excel_dir}")
            else:
                print(f"\nFound {len(excel_files)} Excel files. Analyzing...")
                
                # 存储所有结果
                results = []
                
                # 分析每个文件
                for file_path in excel_files:
                    try:
                        result = get_excel_volume_score(file_path)
                        results.append({
                            'file_name': file_path.name,
                            'size_mb': result['size_mb'],
                            'volume_score': result['volume_score'],
                            'rows': result['file_info'].get('rows', 'N/A'),
                            'columns': result['file_info'].get('columns', 'N/A')
                        })
                        print(f"\nProcessed: {file_path.name}")
                        print(f"Size: {result['size_mb']:.2f} MB")
                        print(f"Volume Score: {result['volume_score']:.2f}")
                        print(f"Rows: {result['file_info'].get('rows', 'N/A')}")
                        print(f"Columns: {result['file_info'].get('columns', 'N/A')}")
                    except Exception as e:
                        print(f"Error processing {file_path.name}: {str(e)}")
                
                # 转换为DataFrame并排序
                df_results = pd.DataFrame(results)
                df_results = df_results.sort_values('volume_score', ascending=False)
                
                # 格式化大小列
                df_results['size_mb'] = df_results['size_mb'].map('{:.2f}'.format)
                
                # 显示汇总结果
                print("\n=== Summary of All Files ===")
                print(df_results.to_string(index=False))
                
                # 显示统计信息
                print("\n=== Statistics ===")
                print(f"Total files analyzed: {len(df_results)}")
                print(f"Average volume score: {df_results['volume_score'].mean():.2f}")
                print(f"Highest volume score: {df_results['volume_score'].max():.2f}")
                print(f"Lowest volume score: {df_results['volume_score'].min():.2f}")
                
                # 绘制可视化图表
                plot_volume_analysis(results)
                
        except Exception as e:
            print(f"Error: {str(e)}")

    # # 测试不同大小的数据集
    # print("\n不同大小的数据集得分示例：")
    # test_sizes = [0.001, 0.01, 0.1, 1, 10, 100]  # 更符合Excel文件大小的范围
    # for size in test_sizes:
    #     score = quantify_volume(size)
    #     print(f"{size:.3f} GB 数据集的容量得分: {score:.4f}")