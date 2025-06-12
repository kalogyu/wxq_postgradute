import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

# 1. 创建示例数据集
np.random.seed(42)
n_samples = 1000

# 创建特征
X1 = np.random.normal(0, 1, n_samples)  # 特征1
X2 = np.random.normal(0, 1, n_samples)  # 特征2
X3 = X1 * 0.5 + np.random.normal(0, 0.5, n_samples)  # 特征3

# 创建目标变量（二分类）
y = (X1 + X2 > 0).astype(int)  # 如果X1+X2>0则为1，否则为0

# 创建DataFrame
df = pd.DataFrame({
    'feature1': X1,
    'feature2': X2,
    'feature3': X3,
    'target': y
})

# 2. 准备数据
X = df[['feature1', 'feature2', 'feature3']]  # 特征
y = df['target']  # 目标变量

# 3. 数据集划分
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,  # 20%的数据作为测试集
    random_state=42  # 设置随机种子，确保结果可重现
)

print("训练集大小:", X_train.shape)
print("测试集大小:", X_test.shape)

# 4. 创建随机森林模型
model = RandomForestClassifier(
    n_estimators=100,  # 使用100棵决策树
    random_state=42    # 设置随机种子
)

# 5. 训练模型
model.fit(X_train, y_train)

# 6. 预测
y_pred = model.predict(X_test)

# 7. 评估模型
score = f1_score(y_test, y_pred, average='weighted')
print("模型F1分数:", score)

# 8. 查看特征重要性
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
})
print("\n特征重要性:")
print(feature_importance.sort_values('importance', ascending=False))