import numpy as np
import pandas as pd
from typing import Dict, Union, List, Tuple
from datetime import datetime

def calculate_missing_rate(df: pd.DataFrame) -> float:
    """计算缺失值比例"""
    return df.isnull().sum().sum() / (df.shape[0] * df.shape[1])

def calculate_duplicate_rate(df: pd.DataFrame) -> float:
    """计算重复行比例"""
    return df.duplicated().sum() / len(df)

def calculate_outlier_rate(df: pd.DataFrame, columns: List[str] = None) -> float:
    """计算异常值比例（使用IQR方法）"""
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    outlier_count = 0
    total_count = 0
    
    for col in columns:
        if df[col].dtype in [np.float64, np.int64]:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
            outlier_count += outliers
            total_count += len(df)
    
    return outlier_count / total_count if total_count > 0 else 0

def calculate_consistency_score(df: pd.DataFrame) -> float:
    """计算数据一致性得分"""
    # 检查数值型列的标准差
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        return 1.0
    
    # 计算每列的标准差与均值的比值（变异系数）
    cv_scores = []
    for col in numeric_cols:
        if df[col].std() != 0 and df[col].mean() != 0:
            cv = df[col].std() / abs(df[col].mean())
            cv_scores.append(1 / (1 + cv))  # 转换为0-1之间的得分
    
    return np.mean(cv_scores) if cv_scores else 1.0

def calculate_data_type_consistency(df: pd.DataFrame) -> float:
    """计算数据类型一致性得分"""
    type_consistency = 0
    total_columns = len(df.columns)
    
    for col in df.columns:
        # 检查列中是否所有值都是相同类型
        if df[col].dtype in [np.float64, np.int64]:
            # 对于数值列，检查是否都是整数或都是浮点数
            if df[col].dtype == np.int64:
                type_consistency += 1 if df[col].apply(lambda x: isinstance(x, (int, np.integer)) or pd.isna(x)).all() else 0
            else:
                type_consistency += 1 if df[col].apply(lambda x: isinstance(x, (float, np.floating)) or pd.isna(x)).all() else 0
        else:
            # 对于非数值列，检查是否都是相同类型
            type_consistency += 1 if df[col].apply(lambda x: isinstance(x, type(df[col].iloc[0])) or pd.isna(x)).all() else 0
    
    return type_consistency / total_columns

def calculate_value_range_consistency(df: pd.DataFrame) -> float:
    """计算数值范围一致性得分"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        return 1.0
    
    range_scores = []
    for col in numeric_cols:
        if df[col].dtype in [np.float64, np.int64]:
            # 检查是否有负值（如果数据不应该有负值）
            has_negative = (df[col] < 0).any()
            # 检查是否超出合理范围（例如，年龄不应该超过150）
            if col.lower() in ['age', '年龄']:
                has_unreasonable = (df[col] > 150).any()
            elif col.lower() in ['height', '身高']:
                has_unreasonable = (df[col] > 300).any()  # 假设单位是厘米
            elif col.lower() in ['weight', '体重']:
                has_unreasonable = (df[col] > 500).any()  # 假设单位是千克
            else:
                has_unreasonable = False
            
            score = 1.0
            if has_negative:
                score *= 0.5
            if has_unreasonable:
                score *= 0.5
            range_scores.append(score)
    
    return np.mean(range_scores) if range_scores else 1.0

def calculate_date_consistency(df: pd.DataFrame) -> float:
    """计算日期一致性得分"""
    date_cols = df.select_dtypes(include=['datetime64']).columns
    if len(date_cols) == 0:
        return 1.0
    
    date_scores = []
    for col in date_cols:
        # 检查日期是否在合理范围内
        min_date = df[col].min()
        max_date = df[col].max()
        current_year = datetime.now().year
        
        # 检查是否包含未来日期
        has_future = (df[col] > datetime.now()).any()
        # 检查是否包含过于久远的日期（例如100年前）
        has_ancient = (df[col] < datetime(current_year - 100, 1, 1)).any()
        
        score = 1.0
        if has_future:
            score *= 0.5
        if has_ancient:
            score *= 0.5
        date_scores.append(score)
    
    return np.mean(date_scores) if date_scores else 1.0

def calculate_categorical_consistency(df: pd.DataFrame) -> float:
    """计算分类数据一致性得分"""
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    if len(categorical_cols) == 0:
        return 1.0
    
    cat_scores = []
    for col in categorical_cols:
        # 计算唯一值比例
        unique_ratio = df[col].nunique() / len(df)
        # 如果唯一值比例过高，可能表示数据不一致
        if unique_ratio > 0.5:  # 可以调整这个阈值
            cat_scores.append(0.5)
        else:
            cat_scores.append(1.0)
    
    return np.mean(cat_scores) if cat_scores else 1.0

def quantify_quality(df: pd.DataFrame, 
                    weights: Dict[str, float] = None) -> Dict[str, Union[float, Dict[str, float]]]:
    """
    全面评估数据集的质量
    
    参数:
        df: pandas DataFrame，要评估的数据集
        weights: 字典，各指标的权重，默认权重为均等
        
    返回:
        包含总体质量得分和各项指标得分的字典
    """
    if weights is None:
        weights = {
            'missing_rate': 0.15,
            'duplicate_rate': 0.15,
            'outlier_rate': 0.15,
            'consistency': 0.15,
            'type_consistency': 0.15,
            'range_consistency': 0.10,
            'date_consistency': 0.05,
            'categorical_consistency': 0.10
        }
    
    # 计算各项指标
    missing_rate = calculate_missing_rate(df)
    duplicate_rate = calculate_duplicate_rate(df)
    outlier_rate = calculate_outlier_rate(df)
    consistency_score = calculate_consistency_score(df)
    type_consistency = calculate_data_type_consistency(df)
    range_consistency = calculate_value_range_consistency(df)
    date_consistency = calculate_date_consistency(df)
    categorical_consistency = calculate_categorical_consistency(df)
    
    # 计算各项指标的得分
    scores = {
        'missing_rate': 1 - missing_rate,
        'duplicate_rate': 1 - duplicate_rate,
        'outlier_rate': 1 - outlier_rate,
        'consistency': consistency_score,
        'type_consistency': type_consistency,
        'range_consistency': range_consistency,
        'date_consistency': date_consistency,
        'categorical_consistency': categorical_consistency
    }
    
    # 计算总体质量得分
    total_score = sum(scores[metric] * weights[metric] for metric in weights)
    
    return {
        'total_score': total_score,
        'detailed_scores': scores,
        'weights': weights
    }
