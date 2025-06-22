from dotenv import load_dotenv
import os
load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))
import numpy as np
import pandas as pd
from typing import Dict, Union, List, Tuple, Optional
from datetime import datetime
import json
import re
import requests

# LLM-based metrics configuration
LLM_METRICS_CONFIG = {
    "value_range": {
        "display_name": "数值范围一致性",
        "description": "适用于有数值型字段的表格，判断数值是否在合理范围内。"
    },
    "date_consistency": {
        "display_name": "日期一致性",
        "description": "适用于有日期/时间字段的表格，判断日期格式和范围是否合理。"
    },
    "categorical_consistency": {
        "display_name": "分类数据一致性",
        "description": "适用于有分类/枚举型字段的表格，判断分类是否规范、无杂项。"
    }
}

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

def _get_prompt_sample(df: pd.DataFrame, max_sample_chars: int = 2000) -> str:
    """
    智能采样并格式化为Markdown表格，保证样例部分不超过max_sample_chars。
    - 若行数<20，全展示；
    - 否则采样min(10%总行数, 200)行；
    - 超长则递减采样量。
    """
    if len(df) < 20:
        sample_df = df
    else:
        sample_size = min(int(len(df) * 0.1), 200)
        if sample_size < 1:
            sample_size = 1
        sample_df = df.sample(n=sample_size, random_state=42)

    sample_str = sample_df.to_markdown(index=False)

    # 动态调整采样量
    while len(sample_str) > max_sample_chars and len(sample_df) > 10:
        sample_df = sample_df.sample(frac=0.8, random_state=42)
        sample_str = sample_df.to_markdown(index=False)

    # 最终截断
    if len(sample_str) > max_sample_chars:
        sample_str = sample_str[:max_sample_chars] + "\n... (表格过长，已截断)"

    return sample_str

class AgentQualityScorer:
    """
    使用大模型（如文心一言）自动评分数据一致性（数值范围、日期、分类）
    """
    def __init__(self, access_token: Optional[str] = None):
        self.api_key = os.getenv('ERNIE_API_KEY')
        self.secret_key = os.getenv('ERNIE_SECRET_KEY')
        self.token_url = "https://aip.baidubce.com/oauth/2.0/token"
        self.access_token = access_token or self._get_access_token()

    def _get_access_token(self) -> Optional[str]:
        """获取access_token"""
        if not self.api_key or not self.secret_key:
            print("未设置ERNIE_API_KEY或ERNIE_SECRET_KEY环境变量")
            return None
        params = {
            'grant_type': 'client_credentials',
            'client_id': self.api_key,
            'client_secret': self.secret_key
        }
        try:
            response = requests.post(self.token_url, params=params)
            response.raise_for_status()
            result = response.json()
            return result.get('access_token', None)
        except Exception as e:
            print(f"获取access_token失败: {e}")
            return None

    def _call_ernie_api(self, prompt: str) -> dict:
        """调用百度文心一言API，详细打印每一步"""
        print("[文心一言] 即将调用API...\n")
        if not self.access_token:
            print("[文心一言] access_token获取失败，无法调用API。\n")
            return {"score": 0.5, "reason": "access_token获取失败", "raw_response": None}
        url = f"https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/completions?access_token={self.access_token}"
        headers = {'Content-Type': 'application/json'}
        payload = {
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7,
            "top_p": 0.8
        }
        print("[文心一言] 发送给API的Prompt内容如下：\n" + "="*40)
        print(prompt)
        print("="*40 + "\n")
        # 日志记录
        log_dir = "logs"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"ernie_api_log_{timestamp}.txt")
        try:
            with open(log_file, "w", encoding="utf-8") as f:
                f.write("=== 发送给文心一言的Prompt ===\n\n")
                f.write(prompt)
                f.write("\n\n=== 文心一言的输出 ===\n\n")
            print("[文心一言] 正在发送请求...\n")
            response = requests.post(url, headers=headers, json=payload)
            print("[文心一言] 已收到响应，正在处理...\n")
            response.raise_for_status()
            result = response.json()
            text = result.get('result', '')
            print("[文心一言] 收到的原始响应内容如下：\n" + "="*40)
            print(text)
            print("="*40 + "\n")
            text = re.sub(r'```json|```', '', text).strip()
            # 追加API输出到日志
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(text)
            print("[文心一言] 正在尝试解析JSON...\n")
            try:
                parsed = json.loads(text)
                print("[文心一言] JSON解析成功！\n")
            except Exception:
                print("[文心一言] JSON解析失败，返回默认分数。\n")
                parsed = {"score": 0.5, "reason": "API返回内容解析失败"}
            parsed['raw_response'] = text
            return parsed
        except Exception as e:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"API调用失败: {e}\n")
            print(f"[文心一言] API调用失败: {e}\n")
            return {"score": 0.5, "reason": "API调用失败，返回默认分数", "raw_response": None}

    def _prepare_prompt(self, df: pd.DataFrame, score_type: str) -> str:
        columns = df.columns.tolist()
        sample = _get_prompt_sample(df)
        prompt = f"""
你是一个资深数据质量专家。请根据下表的内容和统计信息，评估其"{score_type}"得分（0-1之间，1为最佳，0为最差）。
下表为样例数据（Markdown表格格式）：
{sample}

请严格按照以下标准打分，并给出详细理由：

评分标准：
- 1.0：数据完全没有此类问题（如无异常值/日期格式完全一致/分类无杂项），且样本充足。
- 0.8-0.99：仅有极少量轻微问题，不影响整体分析。
- 0.6-0.79：存在一定比例的问题，但大部分数据仍然可靠。
- 0.4-0.59：问题较多，影响部分分析结论。
- 0.2-0.39：问题严重，大部分数据不可靠。
- 0-0.19：几乎全部数据有严重问题，无法用于分析。

请结合以下信息进行判断：
- 表头: {columns}
- 如有必要，可参考缺失值、异常值、格式不一致等情况。

请输出如下JSON格式（不要输出多余内容）：
{{
  "score": 得分,
  "reason": "详细理由，说明具体哪些地方有问题或为何得分高/低"
}}
"""
        return prompt

    def score(self, df: pd.DataFrame, score_type: str) -> dict:
        prompt = self._prepare_prompt(df, score_type)
        return self._call_ernie_api(prompt)

def calculate_llm_quality_scores(df: pd.DataFrame, metrics_to_score: List[str]) -> dict:
    """
    一次性让大模型对所有主观指标评分。
    """
    columns = df.columns.tolist()
    sample = _get_prompt_sample(df)

    metric_desc = [f"{i+1}. {LLM_METRICS_CONFIG[m]['display_name']}：{LLM_METRICS_CONFIG[m]['description']}"
                   for i, m in enumerate(metrics_to_score)]
    metric_json = [f'  "{m}": {{"score": 得分, "reason": "理由"}}' for m in metrics_to_score]
    
    metric_desc_str = "\n".join(metric_desc)
    metric_json_str = ',\n'.join(metric_json)

    prompt = f"""
你是一个资深数据质量专家。请根据下表的内容和统计信息，分别评估以下各项的得分（0-1之间，1为最佳，0为最差），并给出详细理由：

下表为样例数据（Markdown表格格式）：
{sample}

请对以下指标评分:
{metric_desc_str}

评分标准：
- 1.0：数据完全没有此类问题，且样本充足。
- 0.8-0.99：仅有极少量轻微问题，不影响整体分析。
- 0.6-0.79：存在一定比例的问题，但大部分数据仍然可靠。
- 0.4-0.59：问题较多，影响部分分析结论。
- 0.2-0.39：问题严重，大部分数据不可靠。
- 0-0.19：几乎全部数据有严重问题，无法用于分析。

表头: {columns}

请只返回如下JSON格式：
{{
{metric_json_str}
}}
"""
    agent = AgentQualityScorer()
    result = agent._call_ernie_api(prompt)
    return result if isinstance(result, dict) else {}

def recommend_quality_weights(df: pd.DataFrame, scores_for_weighting: dict) -> dict:
    """
    让大模型根据表头、样例数据和各项评分结果，智能推荐各项质量指标的权重。
    """
    columns = df.columns.tolist()
    sample = _get_prompt_sample(df)
    
    active_metrics = list(scores_for_weighting.keys())

    # 评分结果摘要
    llm_score_str = "\n".join([
        f"- {k}: 分数={(v.get('score', 0.5) if isinstance(v, dict) else v):.2f}, 原因={v.get('reason', '无') if isinstance(v, dict) else '自动计算'}"
        for k, v in scores_for_weighting.items()
    ])
    
    # 动态生成指标清单和JSON格式
    metric_list_str = "\n".join([f"- {m}" for m in active_metrics])
    json_format_str = ",\n".join([f'  "{m}": 权重' for m in active_metrics])

    prompt = f"""
你是一个数据质量评估专家。请根据下表的表头、样例数据和各项评分结果，结合数据类型和业务直觉，为以下各项质量指标分配合理的权重（总和为1），并简要说明理由：
下表为样例数据（Markdown表格格式）：
{sample}

各项评分结果:
{llm_score_str}

请为以下指标清单分配权重:
{metric_list_str}

请只返回如下JSON格式：
{{
{json_format_str},
  "reason": "简要说明权重分配的依据"
}}
"""
    agent = AgentQualityScorer()
    result = agent._call_ernie_api(prompt)

    if not isinstance(result, dict):
        return {}
    return result

def quantify_quality(df: pd.DataFrame, 
                    weights: Dict[str, float] = None) -> Dict[str, Union[float, Dict[str, float]]]:
    """
    全面评估数据集的质量
    """
    # 1. 计算各项自动指标
    scores = {
        'missing_rate': 1 - calculate_missing_rate(df),
        'duplicate_rate': 1 - calculate_duplicate_rate(df),
        'outlier_rate': 1 - calculate_outlier_rate(df),
        'type_consistency': calculate_data_type_consistency(df),
    }

    # 2. 决定需要哪些LLM指标
    metric_select = select_quality_metrics(df)
    llm_metrics_to_score = [k for k, v in metric_select.items() if v is True and k in LLM_METRICS_CONFIG]

    # 3. 一次性调用大模型获取LLM指标分数
    llm_scores = calculate_llm_quality_scores(df, llm_metrics_to_score) if llm_metrics_to_score else {}

    # 4. 合并所有分数到一个详细的字典，并标记不适用的指标
    detailed_scores = scores.copy()
    for metric_key in LLM_METRICS_CONFIG.keys():
        if metric_key in llm_metrics_to_score:
            detailed_scores[metric_key] = llm_scores.get(metric_key, {'score': 0.5, 'reason': '评估失败'})
        else:
            # 如果指标未被选中，则标记为不适用，分数为1.0
            detailed_scores[metric_key] = {'score': 1.0, 'reason': '不适用'}
    
    # 5. 确定用于计算总分和权重的活动指标
    active_metrics_keys = list(scores.keys()) + llm_metrics_to_score
    scores_for_weighting = {k: detailed_scores[k] for k in active_metrics_keys}

    # 6. 推荐权重（放到最后，基于活动指标的评分结果）
    if weights is None:
        recommended_weights = recommend_quality_weights(df, scores_for_weighting)
        # 兼容异常情况
        if not recommended_weights or not all(k in recommended_weights for k in active_metrics_keys):
            # 如果推荐失败，为活动指标生成默认权重
            num_active = len(active_metrics_keys)
            default_weight = 1.0 / num_active if num_active > 0 else 0
            weights = {k: default_weight for k in active_metrics_keys}
            weights['reason'] = '推荐失败，使用默认平均权重'
        else:
            weights = recommended_weights

    # 7. 计算最终得分（只使用活动指标）
    final_scores = {k: v.get('score') if isinstance(v, dict) else v for k, v in detailed_scores.items()}
    total_score = sum(final_scores.get(k, 0) * weights.get(k, 0) for k in active_metrics_keys)

    return {
        'total_score': total_score,
        'detailed_scores': detailed_scores,
        'weights': weights,
        'llm_metric_select': metric_select
    }

def select_quality_metrics(df: pd.DataFrame) -> dict:
    """
    让大模型根据表头和样例数据，判断是否需要对数值范围、日期一致性、分类一致性进行评分。
    返回: dict, 形如：
    {
      "value_range": True/False,
      "date_consistency": True/False,
      "categorical_consistency": True/False,
      "reason": "..."
    }
    """
    columns = df.columns.tolist()
    sample = _get_prompt_sample(df)

    metric_desc = [f"{i+1}. {v['display_name']}：{v['description']}" 
                   for i, v in enumerate(LLM_METRICS_CONFIG.values())]
    metric_json = [f'  "{k}": true/false' for k in LLM_METRICS_CONFIG.keys()]
    
    metric_desc_str = "\n".join(metric_desc)
    json_format_str = ",\n".join(metric_json)

    prompt = f"""
你是一个数据分析专家。请根据下表的表头和样例数据，判断是否需要对以下各项进行质量评分，并说明理由：
{metric_desc_str}

下表为样例数据（Markdown表格格式）：
{sample}

请只返回如下JSON格式：
{{
{json_format_str},
  "reason": "简要说明每一项的判断理由"
}}
"""
    agent = AgentQualityScorer()
    result = agent._call_ernie_api(prompt)
    # 兼容异常情况
    if not isinstance(result, dict):
        # 如果API调用失败，默认认为所有指标都适用，以进行后续评分
        return {**{k: True for k in LLM_METRICS_CONFIG.keys()}, "reason": "API异常，默认全部适用"}
    
    for k in LLM_METRICS_CONFIG.keys():
        if k not in result:
            result[k] = True # 如果模型没有返回某个键，默认为适用
    if "reason" not in result:
        result["reason"] = "API未返回理由"
    return result

if __name__ == "__main__":
    import pandas as pd
    file_path = r"E:\wxq_postgradute\数据集\extracted_excel_files\地震监测情况.xlsx"
    try:
        df = pd.read_excel(file_path)
        result = quantify_quality(df)
        print("\n==== 数据质量评估结果 ====")
        print(f"总体质量得分: {result['total_score']:.4f}")
        print("详细得分:")
        for k, v in result['detailed_scores'].items():
            if isinstance(v, dict) and 'score' in v:
                print(f"  {k}: {v['score']:.4f}  原因: {v.get('reason', '无理由')}")
            else:
                print(f"  {k}: {v:.4f}")
        print("权重:")
        for k, v in result['weights'].items():
            if k == 'reason':
                continue
            print(f"  {k}: {v}")
        print(f"权重推荐理由: {result['weights'].get('reason', '无理由')}")
        # 打印详细大模型原始回答
        print("\n==== 大模型详细原始回答 ====")
        # 只打印三项大模型评分的详细内容
        for key in LLM_METRICS_CONFIG.keys():
            detail = result['detailed_scores'].get(key)
            if isinstance(detail, dict) and 'raw_response' in detail:
                print(f"[{key}] 原始回答:\n{detail['raw_response']}\n")
    except Exception as e:
        print(f"数据质量评估失败: {e}")
