
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

# 导入数据
df = pd.read_csv('F:\论文\kaggle\Titanic\\train.csv', header=0)

# 设置y值
X = df.drop(["Survived"], axis=1)
y = df["Survived"]

# 训练集和测试集划分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0, shuffle=True)

# 训练模型
dtree = DecisionTreeClassifier(criterion="entropy", random_state=123,
                               max_depth=4,
                               min_samples_leaf=5)
dtree.fit(X_train, y_train)

# 预测
pred_train = dtree.predict(X_train)
pred_test = dtree.predict(X_test)

# 准确率
train_acc = accuracy_score(y_train, pred_train)
test_acc = accuracy_score(y_test, pred_test)
print("训练集准确率: {0:.2f}, 测试集准确率: {1:.2f}".format(train_acc, test_acc))

# 其他模型评估指标
precision, recall, F1, _ = precision_recall_fscore_support(y_test, pred_test, average="binary")
print("精准率: {0:.2f}. 召回率: {1:.2f}, F1分数: {2:.2f}".format(precision, recall, F1))

# 特征重要度
features = list(X_test.columns)
importances = dtree.feature_importances_
indices = np.argsort(importances)[::-1]
num_features = len(importances)

# 将特征重要度以柱状图展示
plt.figure()
plt.title("Feature importances")
plt.bar(range(num_features), importances[indices], color="g", align="center")
plt.xticks(range(num_features), [features[i] for i in indices], rotation='45')
plt.xlim([-1, num_features])
plt.show()

# 输出各个特征的重要度
for i in indices:
    print("{0} - {1:.3f}".format(features[i], importances[i]))