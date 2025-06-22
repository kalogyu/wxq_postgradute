import numpy as np
import pandas as pd
import logging
from typing import Union, Dict, Optional
import os
from pathlib import Path
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def s_volume_function(V: Union[float, np.ndarray], 
                     V_optimal: float = 1000.0,
                     k: float = 0.002) -> Union[float, np.ndarray]:
    """
    计算数据集容量得分函数 S_Volume(V)
    S_Volume(V) = 100 / (1 + e^(-k * (V - V_optimal)))
    
    Args:
        V: 文件大小（MB）
        V_optimal: 最佳文件大小阈值（MB）
        k: 曲线陡峭度参数
        
    Returns:
        数据量得分 (0-100之间)
    """
    return 100 / (1 + np.exp(-k * (V - V_optimal)))

def calculate_volume_score(data: Union[pd.DataFrame, str],
                         V_optimal: float = 1000.0,
                         k: float = 0.002) -> Dict[str, float]:
    """
    计算数据集的数据量得分
    
    Args:
        data: 数据集（DataFrame或文件路径）
        V_optimal: 最佳文件大小阈值（MB）
        k: 曲线陡峭度参数
        
    Returns:
        包含数据量得分的字典
    """
    try:
        # 获取文件大小（MB）
        if isinstance(data, str):
            if not os.path.exists(data):
                raise FileNotFoundError(f"文件不存在: {data}")
            volume = os.path.getsize(data) / (1024 * 1024)  # 转换为MB
        else:
            volume = data.memory_usage(deep=True).sum() / (1024 * 1024)  # 转换为MB
        
        # 计算得分
        score = s_volume_function(volume, V_optimal, k)
        
        return {
            'volume': volume,
            'score': score
        }
        
    except Exception as e:
        logger.error(f"计算数据量得分时发生错误: {str(e)}")
        raise

def plot_volume_score_curve(V_optimal: float = 1000.0,
                          k: float = 0.002,
                          min_volume: float = 0.0,
                          max_volume: float = 2000.0,
                          save_path: str = "volume_score_curve.png",
                          highlight_files: Optional[list[str]] = None):
    """
    绘制数据量得分曲线，并可选择高亮特定文件位置
    
    Args:
        V_optimal: 最佳文件大小阈值（MB）
        k: 曲线陡峭度参数
        min_volume: 最小文件大小（MB）
        max_volume: 最大文件大小（MB）
        save_path: 图片保存路径
        highlight_files: 要在图上高亮显示的文件路径列表 (可选)
    """
    # 生成文件大小范围
    volumes = np.linspace(min_volume, max_volume, 500)
    
    # 计算对应的得分
    scores = s_volume_function(volumes, V_optimal, k)
    
    # 创建图形
    plt.figure(figsize=(10, 6))
    
    # 绘制主曲线
    plt.plot(volumes, scores, 'b-', label=f'S_Volume(V) with V_optimal={V_optimal}MB, k={k}')
    
    # 添加最佳值点
    plt.axvline(x=V_optimal, color='r', linestyle='--', 
                label=f'V_optimal = {V_optimal}MB (Score = 50)')
    plt.axhline(y=50, color='gray', linestyle=':', label='Score = 50')
    
    # 高亮特定文件
    if highlight_files:
        for file_path in highlight_files:
            if os.path.exists(file_path):
                try:
                    volume_mb = os.path.getsize(file_path) / (1024 * 1024)
                    score = s_volume_function(volume_mb, V_optimal, k)
                    
                    # 在图上标记点
                    plt.scatter(volume_mb, score, color='green', s=50, zorder=5)
                    
                    # 添加注释
                    plt.annotate(os.path.basename(file_path),
                                 xy=(volume_mb, score),
                                 xytext=(5, 5),
                                 textcoords='offset points',
                                 ha='left',
                                 va='bottom',
                                 fontsize=8,
                                 bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.5))
                except Exception as e:
                    logger.error(f"高亮文件 {file_path} 时出错: {e}")

    # 设置坐标轴标签
    plt.xlabel('文件大小 (MB)')
    plt.ylabel('得分 (0-100)')
    plt.title('数据集容量得分函数：文件大小与得分的关系')
    
    # 添加网格
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 添加图例
    plt.legend()
    
    # 设置坐标轴范围
    plt.ylim(0, 100)
    
    # 保存图片
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"得分曲线已保存至: {save_path}")

def auto_configure_parameters(file_sizes_mb: list[float]) -> Dict[str, float]:
    """
    根据文件大小分布自动配置V_optimal和k参数。
    调整后的逻辑，使分数分布更合理。

    Args:
        file_sizes_mb (list[float]): 文件大小列表 (MB)

    Returns:
        Dict[str, float]: 包含'V_optimal'和'k'的字典
    """
    if not file_sizes_mb or len(file_sizes_mb) < 2:
        return {'V_optimal': 1000.0, 'k': 0.002} # 返回默认值

    sizes = np.array(file_sizes_mb)
    
    # 1. V_optimal设置为中位数 (50%分位数), 这是S曲线的拐点，得分50.
    #    这比使用80%分位数更能代表数据集的中心趋势。
    v_optimal = np.median(sizes)
    
    # 2. k的计算：使95%分位数的文件得分在95分左右。
    #    这提供了更好的分数分布，而不是仅仅将最大值推到95分。
    v_95 = np.percentile(sizes, 95)
    
    # 避免v_95和v_optimal相等或过于接近导致k值过大或除零
    if v_95 <= v_optimal:
        # 如果分布非常集中，我们使用最大值来计算k，以确保有分数差异
        v_max = np.max(sizes)
        if v_max > v_optimal:
            # ln(19)来自于解 95 = 100 / (1 + exp(-k * (V_max - V_optimal)))
            k = np.log(19) / (v_max - v_optimal)
        else:
            # 如果所有值都几乎相同，则k可以设置为一个较小的值
            # 基于标准差的启发式方法可能适用
            k = 1.0 / (np.std(sizes) + 1e-6) if np.std(sizes) > 0 else 0.1
    else:
        # k = ln(19) / (V_95 - V_optimal)  解: 95 = 100 / (1+exp(-k*(V_95-V_optimal)))
        k = np.log(19) / (v_95 - v_optimal)

    # 限制k的范围，防止过大或过小
    k = np.clip(k, 1e-4, 2.0)
    
    # 确保v_optimal不是0
    if v_optimal == 0:
        v_optimal = np.mean(sizes) if np.mean(sizes) > 0 else 1.0

    return {'V_optimal': v_optimal, 'k': k}

if __name__ == '__main__':
    # 定义要测试的目录
    excel_dir = r"E:\wxq_postgradute\数据集\extracted_excel_files"

    # 扫描目录下的所有.xlsx文件
    try:
        excel_files = list(Path(excel_dir).glob('*.xlsx'))
        if not excel_files:
            print(f"在目录 {excel_dir} 中未找到.xlsx文件。")
    except Exception as e:
        print(f"扫描目录时出错: {e}")
        excel_files = []

    # 如果有文件，则进行分析
    if excel_files:
        # 首先获取所有文件大小以自动配置参数
        file_sizes = [os.path.getsize(str(f)) / (1024 * 1024) for f in excel_files]
        
        # 自动配置参数
        params = auto_configure_parameters(file_sizes)
        V_optimal_auto = params['V_optimal']
        k_auto = params['k']
        
        print("\n=== 自动配置参数 ===")
        print(f"自动配置的 V_optimal: {V_optimal_auto:.2f} MB")
        print(f"自动配置的 k        : {k_auto:.4f}")

        # 计算每个文件的得分
        results = []
        print(f"\n找到 {len(excel_files)} 个Excel文件。正在使用自动参数进行分析...")
        for i, file_path in enumerate(excel_files):
            try:
                # 使用自动配置的参数计算得分
                score = s_volume_function(file_sizes[i], V_optimal_auto, k_auto)
                results.append({
                    'file_name': os.path.basename(file_path),
                    'volume_mb': file_sizes[i],
                    'score': score
                })
            except Exception as e:
                print(f"处理文件 {file_path} 时出错: {e}")

        # 显示结果摘要
        if results:
            df_results = pd.DataFrame(results).sort_values('score', ascending=False).reset_index(drop=True)
            # 格式化输出
            df_results['volume_mb'] = df_results['volume_mb'].map('{:.2f}'.format)
            df_results['score'] = df_results['score'].map('{:.2f}'.format)
            print("\n=== 所有Excel文件的容量得分分析 (使用自动参数) ===")
            print(df_results.to_string())

        # 绘制得分曲线并高亮所有文件
        # 为了让曲线更具可读性，我们可以将max_volume设置为文件最大值的1.2倍
        max_plot_volume = max(file_sizes) * 1.2 if file_sizes else 2000
        plot_volume_score_curve(
            V_optimal=V_optimal_auto,
            k=k_auto,
            min_volume=0.0,
            max_volume=max_plot_volume,
            save_path="volume.png",
            highlight_files=[str(f) for f in excel_files]
        )
        
        print(f"\n已生成得分曲线图 'volume.png'，并高亮了 {len(excel_files)} 个文件。")
    else:
        # 如果没有文件，仍然可以画一个默认的曲线
        plot_volume_score_curve()
        print("\n未找到文件，已生成默认得分曲线图 'volume.png'。")
    