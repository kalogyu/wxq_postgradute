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
                          save_path: str = "volume_score_curve.png"):
    """
    绘制数据量得分曲线
    
    Args:
        V_optimal: 最佳文件大小阈值（MB）
        k: 曲线陡峭度参数
        min_volume: 最小文件大小（MB）
        max_volume: 最大文件大小（MB）
        save_path: 图片保存路径
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

if __name__ == '__main__':
    plot_volume_score_curve(
        V_optimal=5000.0,  # 最佳文件大小阈值（MB）
        k=0.002,
        min_volume=0.0,
        max_volume=15000.0,
        save_path="volume.png"
    ) 