# 数据集质量评估工具

这个项目提供了一套工具来评估数据集的质量，包括数据质量、隐私保护和机器学习效果等多个维度。

## 功能特点

- 数据质量评估
- 隐私保护评估
- 机器学习效果评估
- 多维度综合评分

## 安装要求

- Python 3.8+
- 依赖包：见 requirements.txt

## 安装步骤

1. 克隆仓库：
```bash
git clone [repository_url]
cd [repository_name]
```

2. 创建虚拟环境（推荐）：
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate
```

3. 安装依赖：
```bash
pip install -r requirements.txt
```

## 使用示例

```python
import pandas as pd
from utils.get_Quality import quantify_quality
from utils.get_Privacy import quantify_privacy_protection
from utils.get_ML_Prediction import quantify_ml_effect

# 加载数据
df = pd.read_csv('your_data.csv')

# 评估数据质量
quality_result = quantify_quality(df)
print("数据质量得分:", quality_result['total_score'])

# 评估隐私保护
privacy_result = quantify_privacy_protection(
    anonymization_level="pseudonymized",
    compliance_status="partial_compliant",
    data_types_in_dataset=["name", "phone"],
    incident_history_level="low_risk"
)
print("隐私保护得分:", privacy_result['total_score'])

# 评估机器学习效果
ml_result = quantify_ml_effect(
    dataset_df=df,
    target_column='target',
    task_type='classification'
)
print("机器学习效果得分:", ml_result['total_score'])
```

## 项目结构

```
project/
├── main.py              # 主程序入口
├── requirements.txt     # 依赖包列表
├── README.md           # 项目说明文档
└── utils/              # 工具函数目录
    ├── __init__.py
    ├── get_Quality.py
    ├── get_Privacy.py
    └── get_ML_Prediction.py
```

## 注意事项

- 确保数据格式正确
- 根据实际需求调整评估参数
- 注意数据隐私保护

## 贡献指南

欢迎提交 Issue 和 Pull Request

## 许可证

MIT License 