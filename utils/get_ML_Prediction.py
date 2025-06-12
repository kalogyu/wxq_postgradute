import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from typing import Dict, Union, List, Optional, Tuple
# from sklearn.linear_model import LogisticRegression, LinearRegression # 可以根据需求选择模型

def calculate_feature_importance(X: pd.DataFrame, y: pd.Series, task_type: str) -> float:
    """
    计算特征重要性得分
    
    参数:
        X: 特征数据
        y: 目标变量
        task_type: 任务类型 ('classification' 或 'regression')
    
    返回:
        float: 特征重要性得分 (0-1)
    """
    if task_type == 'classification':
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    else:
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    
    model.fit(X, y)
    importances = model.feature_importances_
    return np.mean(importances)  # 返回平均特征重要性

def calculate_class_balance(y: pd.Series) -> float:
    """
    计算类别平衡性得分
    
    参数:
        y: 目标变量
    
    返回:
        float: 类别平衡性得分 (0-1)
    """
    class_counts = y.value_counts()
    n_classes = len(class_counts)
    if n_classes == 1:
        return 0.0  # 只有一个类别，完全不平衡
    
    # 计算类别分布的熵
    probs = class_counts / len(y)
    entropy = -np.sum(probs * np.log2(probs))
    max_entropy = np.log2(n_classes)
    
    return entropy / max_entropy  # 归一化到0-1

def calculate_feature_correlation(X: pd.DataFrame) -> float:
    """
    计算特征相关性得分
    
    参数:
        X: 特征数据
    
    返回:
        float: 特征相关性得分 (0-1)
    """
    corr_matrix = X.corr().abs()
    # 移除对角线
    np.fill_diagonal(corr_matrix.values, 0)
    # 计算平均相关性
    mean_corr = corr_matrix.values.mean()
    return 1 - mean_corr  # 相关性越低越好

def calculate_model_performance(X: pd.DataFrame, y: pd.Series, task_type: str) -> float:
    """
    计算模型性能得分
    
    参数:
        X: 特征数据
        y: 目标变量
        task_type: 任务类型
    
    返回:
        float: 模型性能得分 (0-1)
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    if task_type == 'classification':
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        score = f1_score(y_test, y_pred, average='weighted')
    else:
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        score = r2_score(y_test, y_pred)
        # 将R2分数映射到0-1范围
        score = max(0, min(1, score))
    
    return score

def quantify_ml_effect(
    dataset_df: pd.DataFrame,
    target_column: str,
    task_type: str = 'classification',
    weights: Optional[Dict[str, float]] = None
) -> Dict[str, Union[float, Dict[str, float]]]:
    """
    评估数据集在机器学习任务上的预测效果
    
    参数:
        dataset_df: 输入数据集
        target_column: 目标变量列名
        task_type: 任务类型 ('classification' 或 'regression')
        weights: 各评估维度的权重
    
    返回:
        Dict: 包含总体得分和详细得分的字典
    """
    if target_column not in dataset_df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset")
    
    # 准备数据
    X = dataset_df.drop(columns=[target_column])
    y = dataset_df[target_column]
    
    # 数据预处理
    X = pd.get_dummies(X)  # 处理分类特征
    X = X.fillna(X.mean())  # 处理缺失值
    
    # 标准化数值特征
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    
    # 默认权重
    if weights is None:
        weights = {
            'model_performance': 0.4,
            'feature_importance': 0.3,
            'class_balance': 0.2,
            'feature_correlation': 0.1
        }
    
    # 计算各项得分
    model_score = calculate_model_performance(X_scaled, y, task_type)
    feature_importance_score = calculate_feature_importance(X_scaled, y, task_type)
    class_balance_score = calculate_class_balance(y) if task_type == 'classification' else 1.0
    feature_correlation_score = calculate_feature_correlation(X_scaled)
    
    # 计算加权总分
    total_score = (
        model_score * weights['model_performance'] +
        feature_importance_score * weights['feature_importance'] +
        class_balance_score * weights['class_balance'] +
        feature_correlation_score * weights['feature_correlation']
    )
    
    return {
        'total_score': total_score,
        'detailed_scores': {
            'model_performance': model_score,
            'feature_importance': feature_importance_score,
            'class_balance': class_balance_score,
            'feature_correlation': feature_correlation_score
        },
        'weights': weights
    }