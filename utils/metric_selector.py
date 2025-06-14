import numpy as np
from typing import List, Dict, Any
from dataclasses import dataclass
import json
import os
from dotenv import load_dotenv
import requests
import time
import logging
import re
import pandas as pd
from pathlib import Path

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 加载环境变量
load_dotenv()

@dataclass
class DatasetCharacteristics:
    """Class to store dataset characteristics"""
    num_samples: int
    num_features: int
    data_type: str  # 'image', 'text', 'tabular', 'time_series'
    has_labels: bool
    is_balanced: bool
    has_missing_values: bool
    has_categorical_features: bool
    has_numerical_features: bool
    has_temporal_features: bool
    has_spatial_features: bool
    has_sensitive_attributes: bool
    data_quality_issues: List[str]

def analyze_dataset(file_path: str) -> DatasetCharacteristics:
    """分析数据集特征"""
    try:
        # 读取Excel文件
        logger.info(f"正在读取文件: {file_path}")
        df = pd.read_excel(file_path)
        
        # 基本特征
        num_samples = len(df)
        num_features = len(df.columns)
        
        # 检查数据类型
        data_type = 'tabular'
        
        # 检查是否有标签（假设最后一列是标签）
        has_labels = True
        
        # 检查数据平衡性（如果是分类问题）
        is_balanced = True
        if df.iloc[:, -1].dtype == 'object':
            value_counts = df.iloc[:, -1].value_counts()
            is_balanced = (value_counts.max() / value_counts.min()) < 2
        
        # 检查缺失值
        has_missing_values = df.isnull().any().any()
        
        # 检查特征类型
        categorical_features = df.select_dtypes(include=['object', 'category']).columns
        numerical_features = df.select_dtypes(include=['int64', 'float64']).columns
        has_categorical_features = len(categorical_features) > 0
        has_numerical_features = len(numerical_features) > 0
        
        # 检查时间特征
        temporal_columns = [col for col in df.columns if any(keyword in col.lower() 
                          for keyword in ['time', 'date', 'year', 'month', 'day'])]
        has_temporal_features = len(temporal_columns) > 0
        
        # 检查空间特征
        spatial_columns = [col for col in df.columns if any(keyword in col.lower() 
                          for keyword in ['longitude', 'latitude', 'location', 'position'])]
        has_spatial_features = len(spatial_columns) > 0
        
        # 检查敏感属性
        sensitive_columns = [col for col in df.columns if any(keyword in col.lower() 
                           for keyword in ['id', 'name', 'address', 'phone', 'email'])]
        has_sensitive_attributes = len(sensitive_columns) > 0
        
        # 检查数据质量问题
        data_quality_issues = []
        if has_missing_values:
            data_quality_issues.append('missing_values')
        
        # 检查异常值（针对数值列）
        for col in numerical_features:
            if df[col].dtype in ['int64', 'float64']:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).any()
                if outliers:
                    data_quality_issues.append('outliers')
                    break
        
        # 创建特征对象
        characteristics = DatasetCharacteristics(
            num_samples=num_samples,
            num_features=num_features,
            data_type=data_type,
            has_labels=has_labels,
            is_balanced=is_balanced,
            has_missing_values=has_missing_values,
            has_categorical_features=has_categorical_features,
            has_numerical_features=has_numerical_features,
            has_temporal_features=has_temporal_features,
            has_spatial_features=has_spatial_features,
            has_sensitive_attributes=has_sensitive_attributes,
            data_quality_issues=data_quality_issues
        )
        
        logger.info("数据集分析完成")
        return characteristics
        
    except Exception as e:
        logger.error(f"分析数据集时发生错误: {str(e)}")
        raise

class MetricSelector:
    """AI Agent for selecting appropriate metrics based on dataset characteristics using Baidu ERNIE Bot API"""
    
    def __init__(self):
        # 设置百度文心一言API密钥
        self.api_key = os.getenv('ERNIE_API_KEY')
        self.secret_key = os.getenv('ERNIE_SECRET_KEY')
        
        # 检查环境变量
        if not self.api_key:
            logger.error("未找到ERNIE_API_KEY环境变量")
            raise ValueError("请设置ERNIE_API_KEY环境变量")
        if not self.secret_key:
            logger.error("未找到ERNIE_SECRET_KEY环境变量")
            raise ValueError("请设置ERNIE_SECRET_KEY环境变量")
            
        logger.info("成功加载API密钥")
        
        self.metrics = {
            'volume': {
                'name': 'Volume Metric',
                'description': 'Measures the size and complexity of the dataset'
            },
            'ml_prediction': {
                'name': 'ML Prediction Metric',
                'description': 'Evaluates dataset suitability for machine learning'
            },
            'privacy': {
                'name': 'Privacy Metric',
                'description': 'Assesses data privacy and security aspects'
            },
            'quality': {
                'name': 'Quality Metric',
                'description': 'Evaluates overall data quality'
            }
        }
        
        # 获取访问令牌
        try:
            self.access_token = self._get_access_token()
            logger.info("成功获取访问令牌")
        except Exception as e:
            logger.error(f"获取访问令牌失败: {str(e)}")
            raise

    def _get_access_token(self) -> str:
        """获取百度文心一言API的访问令牌"""
        url = "https://aip.baidubce.com/oauth/2.0/token"
        params = {
            "grant_type": "client_credentials",
            "client_id": self.api_key,
            "client_secret": self.secret_key
        }
        
        try:
            logger.info("正在请求访问令牌...")
            response = requests.post(url, params=params)
            response.raise_for_status()  # 检查HTTP错误
            result = response.json()
            
            if 'error' in result:
                logger.error(f"获取访问令牌失败: {result['error']}")
                raise Exception(f"API错误: {result['error']}")
                
            token = result.get("access_token")
            if not token:
                logger.error("响应中没有访问令牌")
                raise Exception("未获取到访问令牌")
                
            return token
            
        except requests.exceptions.RequestException as e:
            logger.error(f"请求失败: {str(e)}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"解析响应失败: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"获取访问令牌时发生未知错误: {str(e)}")
            raise

    def _prepare_prompt(self, characteristics: DatasetCharacteristics) -> str:
        """准备发送给文心一言的提示"""
        try:
            # 读取Excel文件获取列名和第一行数据
            file_path = r"E:\wxq_postgradute\数据集\extracted_excel_files\地震监测情况.xlsx"
            df = pd.read_excel(file_path)
            columns = df.columns.tolist()
            first_row = df.iloc[0].tolist()
            
            # 构建数据预览信息
            data_preview = "数据预览：\n"
            data_preview += "列名：" + ", ".join(columns) + "\n"
            data_preview += "第一行数据：" + ", ".join(str(x) for x in first_row)
            
            prompt = f"""作为一个数据科学专家，请分析以下数据集特征，并推荐适用的评估指标。我们有以下四个评估指标函数：

必需指标（必须包含在推荐中）：
1. Volume Metric (get_Volume.py)
   - 用途：评估数据集的规模和复杂度
   - 具体实现：
     * 计算数据集大小（GB）
     * 使用对数函数计算容量得分（0-100分）
     * 考虑最大相关大小（默认0.17GB）
     * 生成文件大小与得分的可视化分析
   - 适用场景：需要评估数据集大小是否合适，以及数据规模对存储和处理的影响

2. Quality Metric (get_Quality.py)
   - 用途：评估数据集的质量
   - 具体实现：
     * 计算缺失值比例
     * 检测重复数据
     * 识别异常值
     * 评估数据一致性
     * 检查数据类型一致性
     * 验证数值范围合理性
     * 检查日期格式一致性
     * 评估分类数据一致性
     * 权重分配：缺失值(15%)、重复数据(15%)、异常值(15%)、一致性(15%)、类型一致性(15%)、范围一致性(10%)、日期一致性(5%)、分类一致性(10%)
   - 适用场景：需要全面评估数据质量，包括完整性、准确性和一致性

可选指标（根据数据集特征决定是否需要）：
3. ML Prediction Metric (get_ML_Prediction.py)
   - 用途：评估数据集对机器学习任务的适用性
   - 具体实现：
     * 使用随机森林评估特征重要性
     * 计算模型性能得分（分类/回归）
     * 评估类别平衡性
     * 分析特征相关性
     * 权重分配：模型性能(40%)、特征重要性(30%)、类别平衡(20%)、特征相关性(10%)
   - 适用场景：数据集将用于机器学习任务，需要评估其预测能力

4. Privacy Metric (get_Privacy.py)
   - 用途：评估数据集的隐私保护程度
   - 具体实现：
     * 评估匿名化等级（原始数据/假名化/差分隐私/完全匿名）
     * 检查合规性状态
     * 分析敏感数据类型（姓名/ID/电话/地址/健康数据/金融数据/位置/生物特征）
     * 评估隐私事件历史风险
     * 权重分配：匿名化(40%)、合规性(30%)、敏感性(20%)、风险(10%)
   - 适用场景：数据集包含敏感信息，需要评估隐私保护措施

当前数据集特征：
- 样本数量: {characteristics.num_samples}
- 特征数量: {characteristics.num_features}
- 数据类型: {characteristics.data_type}
- 是否有标签: {characteristics.has_labels}
- 是否平衡: {characteristics.is_balanced}
- 是否有缺失值: {characteristics.has_missing_values}
- 是否有类别特征: {characteristics.has_categorical_features}
- 是否有数值特征: {characteristics.has_numerical_features}
- 是否有时间特征: {characteristics.has_temporal_features}
- 是否有空间特征: {characteristics.has_spatial_features}
- 是否有敏感属性: {characteristics.has_sensitive_attributes}
- 数据质量问题: {', '.join(characteristics.data_quality_issues)}

{data_preview}

请根据以上信息，分析这个数据集最适合使用哪些评估指标函数。注意：
1. Volume Metric 和 Quality Metric 是必需的，必须包含在推荐中
2. ML Prediction Metric 和 Privacy Metric 是可选的，请根据数据集特征决定是否需要
3. 对于每个推荐的指标，请提供详细的理由，说明为什么这个指标函数适合这个数据集

请直接返回JSON格式的结果，不要包含任何markdown标记，格式如下：
{{
    "recommended_metrics": [
        {{
            "metric": "指标名称（必须是以下之一：Volume Metric, ML Prediction Metric, Privacy Metric, Quality Metric）",
            "reason": "推荐理由（请详细说明为什么这个指标函数适合这个数据集）"
        }}
    ]
}}"""
            return prompt
            
        except Exception as e:
            logger.error(f"准备提示时发生错误: {str(e)}")
            raise

    def _call_ernie_api(self, prompt: str) -> Dict:
        """调用百度文心一言API"""
        url = f"https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/completions?access_token={self.access_token}"
        
        headers = {
            'Content-Type': 'application/json'
        }
        
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.7,
            "top_p": 0.8
        }
        
        try:
            logger.info("正在调用文心一言API...")
            
            # 记录发送的prompt
            log_dir = "logs"
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            log_file = os.path.join(log_dir, f"ernie_api_log_{timestamp}.txt")
            
            with open(log_file, "w", encoding="utf-8") as f:
                f.write("=== 发送给文心一言的Prompt ===\n\n")
                f.write(prompt)
                f.write("\n\n=== 文心一言的输出 ===\n\n")
            
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()  # 检查HTTP错误
            
            result = response.json()
            logger.info("成功获取API响应")
            
            # 记录文心一言的实际输出
            if 'result' in result:
                with open(log_file, "a", encoding="utf-8") as f:
                    f.write(result['result'])
            else:
                with open(log_file, "a", encoding="utf-8") as f:
                    f.write("API返回错误，未获取到文心一言的输出")
            
            if 'error_code' in result:
                logger.error(f"API返回错误: {result['error_code']} - {result.get('error_msg', '未知错误')}")
                raise Exception(f"API错误: {result['error_code']} - {result.get('error_msg', '未知错误')}")
                
            return result
            
        except requests.exceptions.RequestException as e:
            logger.error(f"API请求失败: {str(e)}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"解析API响应失败: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"调用API时发生未知错误: {str(e)}")
            raise

    def _extract_json_from_response(self, text: str) -> Dict:
        """从响应文本中提取JSON内容"""
        # 移除可能的markdown代码块标记
        text = re.sub(r'```json\s*', '', text)
        text = re.sub(r'```\s*$', '', text)
        text = text.strip()
        
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            logger.error(f"JSON解析失败: {str(e)}")
            logger.error(f"处理后的文本: {text}")
            raise

    def _find_metric_key(self, metric_name: str) -> str:
        """根据指标名称查找对应的key"""
        # 创建反向映射
        name_to_key = {v['name']: k for k, v in self.metrics.items()}
        
        # 尝试直接匹配
        if metric_name in name_to_key:
            return name_to_key[metric_name]
            
        # 尝试模糊匹配
        for name, key in name_to_key.items():
            if name.lower() in metric_name.lower() or metric_name.lower() in name.lower():
                return key
                
        logger.warning(f"未找到匹配的指标: {metric_name}")
        return None

    def select_metrics(self, characteristics: DatasetCharacteristics) -> Dict[str, Any]:
        """
        使用百度文心一言API选择适当的指标
        
        Args:
            characteristics: DatasetCharacteristics对象，包含数据集信息
            
        Returns:
            包含推荐指标和解释的字典
        """
        try:
            # 准备提示
            logger.info("正在准备提示...")
            prompt = self._prepare_prompt(characteristics)
            
            # 调用API
            response = self._call_ernie_api(prompt)
            
            # 解析响应
            if 'result' in response:
                try:
                    result = self._extract_json_from_response(response['result'])
                    logger.info("成功解析API响应")
                    logger.info(f"API返回的指标: {json.dumps(result, ensure_ascii=False, indent=2)}")
                except json.JSONDecodeError as e:
                    logger.error(f"解析结果JSON失败: {str(e)}")
                    logger.error(f"原始响应: {response['result']}")
                    return {}
            else:
                logger.error(f"API响应中没有result字段: {response}")
                return {}
            
            # 转换为所需的输出格式
            recommended_metrics = {}
            for rec in result['recommended_metrics']:
                metric_key = self._find_metric_key(rec['metric'])
                if metric_key:
                    recommended_metrics[metric_key] = {
                        'name': self.metrics[metric_key]['name'],
                        'description': self.metrics[metric_key]['description'],
                        'reason': rec['reason']
                    }
                    logger.info(f"成功匹配指标: {rec['metric']} -> {metric_key}")
                else:
                    logger.warning(f"无法匹配指标: {rec['metric']}")
            
            logger.info(f"成功推荐了 {len(recommended_metrics)} 个指标")
            return recommended_metrics
            
        except Exception as e:
            logger.error(f"选择指标时发生错误: {str(e)}")
            return {}

def get_recommended_metrics(characteristics: DatasetCharacteristics) -> Dict[str, Any]:
    """
    获取数据集推荐的评估指标
    
    Args:
        characteristics: DatasetCharacteristics对象，包含数据集信息
        
    Returns:
        包含推荐指标和解释的字典
    """
    try:
        selector = MetricSelector()
        return selector.select_metrics(characteristics)
    except Exception as e:
        logger.error(f"创建MetricSelector实例时发生错误: {str(e)}")
        return {}

# 示例用法
if __name__ == "__main__":
    try:
        # 设置文件路径
        file_path = r"E:\wxq_postgradute\数据集\extracted_excel_files\地震监测情况.xlsx"
        
        # 分析数据集
        characteristics = analyze_dataset(file_path)
        
        # 获取推荐的指标
        recommended = get_recommended_metrics(characteristics)
        
        # 打印结果
        print("\n数据集特征:")
        print(f"样本数量: {characteristics.num_samples}")
        print(f"特征数量: {characteristics.num_features}")
        print(f"数据类型: {characteristics.data_type}")
        print(f"是否有标签: {characteristics.has_labels}")
        print(f"是否平衡: {characteristics.is_balanced}")
        print(f"是否有缺失值: {characteristics.has_missing_values}")
        print(f"是否有类别特征: {characteristics.has_categorical_features}")
        print(f"是否有数值特征: {characteristics.has_numerical_features}")
        print(f"是否有时间特征: {characteristics.has_temporal_features}")
        print(f"是否有空间特征: {characteristics.has_spatial_features}")
        print(f"是否有敏感属性: {characteristics.has_sensitive_attributes}")
        print(f"数据质量问题: {', '.join(characteristics.data_quality_issues)}")
        
        print("\n推荐的评估指标:")
        for metric, info in recommended.items():
            print(f"\n{info['name']}:")
            print(f"描述: {info['description']}")
            print(f"推荐理由: {info['reason']}")
            
    except Exception as e:
        logger.error(f"运行示例时发生错误: {str(e)}") 