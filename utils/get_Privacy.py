from typing import Dict, List, Union, Optional
import numpy as np

def calculate_anonymization_score(anonymization_level: str) -> float:
    """
    计算匿名化得分
    
    参数:
        anonymization_level: 匿名化等级
            - "raw_data": 原始数据
            - "pseudonymized": 假名化
            - "differential_privacy": 差分隐私
            - "fully_anonymous": 完全匿名化
    
    返回:
        float: 匿名化得分 (0-1)
    """
    anonymization_scores = {
        "raw_data": 0.0,
        "pseudonymized": 0.6,
        "differential_privacy": 0.8,
        "fully_anonymous": 1.0
    }
    return anonymization_scores.get(anonymization_level.lower(), 0.0)

def calculate_compliance_score(compliance_status: str) -> float:
    """
    计算合规性得分
    
    参数:
        compliance_status: 合规性状态
            - "non_compliant": 不合规
            - "partial_compliant": 部分合规
            - "fully_compliant": 完全合规
    
    返回:
        float: 合规性得分 (0-1)
    """
    compliance_scores = {
        "non_compliant": 0.0,
        "partial_compliant": 0.6,
        "fully_compliant": 1.0
    }
    return compliance_scores.get(compliance_status.lower(), 0.0)

def calculate_sensitivity_score(data_types: List[str]) -> float:
    """
    计算数据敏感性得分
    
    参数:
        data_types: 数据集中包含的敏感信息类别列表
            可能的类型包括：
            - "name": 姓名
            - "id": 身份证号
            - "phone": 电话号码
            - "address": 地址
            - "health_data": 健康数据
            - "financial_data": 金融数据
            - "location": 位置信息
            - "biometric": 生物特征
    
    返回:
        float: 敏感性得分 (0-1)
    """
    sensitivity_weights = {
        "name": 0.3,
        "id": 0.4,
        "phone": 0.3,
        "address": 0.3,
        "health_data": 0.5,
        "financial_data": 0.5,
        "location": 0.4,
        "biometric": 0.6
    }
    
    if not data_types:
        return 1.0  # 如果没有敏感数据，得分为1
    
    # 计算所有敏感数据类型的平均权重
    total_weight = sum(sensitivity_weights.get(dtype.lower(), 0.0) for dtype in data_types)
    return 1.0 - (total_weight / len(data_types))  # 转换为0-1得分，权重越高得分越低

def calculate_risk_score(risk_level: str) -> float:
    """
    计算风险评估得分
    
    参数:
        risk_level: 风险等级
            - "high_risk": 高风险
            - "medium_risk": 中等风险
            - "low_risk": 低风险
    
    返回:
        float: 风险得分 (0-1)
    """
    risk_scores = {
        "high_risk": 0.0,
        "medium_risk": 0.5,
        "low_risk": 1.0
    }
    return risk_scores.get(risk_level.lower(), 0.0)

def quantify_privacy_protection(
    anonymization_level: str,
    compliance_status: str,
    data_types_in_dataset: List[str],
    incident_history_level: str,
    weights: Optional[Dict[str, float]] = None
) -> Dict[str, Union[float, Dict[str, float]]]:
    """
    计算数据集的综合隐私保护得分。

    参数:
        anonymization_level: 匿名化/假名化等级
        compliance_status: 合规性状态
        data_types_in_dataset: 数据集中包含的敏感信息类别
        incident_history_level: 隐私事件历史/风险评估等级
        weights: 各子维度的权重字典

    返回:
        Dict: 包含总体隐私保护得分和详细得分的字典
    """
    # 默认权重
    default_weights = {
        'anonymization': 0.40,
        'compliance': 0.30,
        'sensitivity': 0.20,
        'risk': 0.10,
    }

    if weights is None:
        weights = default_weights

    # 确保权重总和为1
    total_weight = sum(weights.values())
    if abs(total_weight - 1.0) > 1e-9:
        normalized_weights = {k: v / total_weight for k, v in weights.items()}
    else:
        normalized_weights = weights

    # 计算各项子维度得分
    score_anonymization = calculate_anonymization_score(anonymization_level)
    score_compliance = calculate_compliance_score(compliance_status)
    score_sensitivity = calculate_sensitivity_score(data_types_in_dataset)
    score_risk = calculate_risk_score(incident_history_level)

    # 计算综合加权得分
    total_score = (
        score_anonymization * normalized_weights['anonymization'] +
        score_compliance * normalized_weights['compliance'] +
        score_sensitivity * normalized_weights['sensitivity'] +
        score_risk * normalized_weights['risk']
    )

    # 返回详细结果
    return {
        'total_score': total_score,
        'detailed_scores': {
            'anonymization': score_anonymization,
            'compliance': score_compliance,
            'sensitivity': score_sensitivity,
            'risk': score_risk
        },
        'weights': normalized_weights
    }