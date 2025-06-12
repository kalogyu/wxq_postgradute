import pandas as pd
import numpy as np
from get_ML_Prediction import quantify_ml_effect

# 创建示例数据集
np.random.seed(42)
n_samples = 1000

# 创建特征
X1 = np.random.normal(0, 1, n_samples)  # 特征1
X2 = np.random.normal(0, 1, n_samples)  # 特征2
X3 = X1 * 0.5 + np.random.normal(0, 0.5, n_samples)  # 与X1相关的特征

# 创建目标变量（分类任务）
y_class = (X1 + X2 > 0).astype(int)  # 二分类目标

# 创建DataFrame
df = pd.DataFrame({
    'feature1': X1,
    'feature2': X2,
    'feature3': X3,
    'target': y_class
})

# 评估数据集
result = quantify_ml_effect(
    dataset_df=df,
    target_column='target',
    task_type='classification'
)

# 打印结果
print("总体得分:", result['total_score'])
print("\n详细得分:")
for metric, score in result['detailed_scores'].items():
    print(f"{metric}: {score:.3f}")