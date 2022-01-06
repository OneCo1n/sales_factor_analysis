
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.datasets import load_breast_cancer
# 二分类解决乳腺癌
cancer = load_breast_cancer()
x = cancer.data
y = cancer.target
train_x, valid_x, train_y, valid_y = train_test_split(x, y, test_size=0.333, random_state=0)  # 分训练集和验证集
# 这里不需要Dmatrix

xlf = xgb.XGBClassifier(max_depth=10,
                        learning_rate=0.01,
                        n_estimators=2000,
                        silent=True,
                        objective='binary:logistic',
                        nthread=-1,
                        gamma=0,
                        min_child_weight=1,
                        max_delta_step=0,
                        subsample=0.85,
                        colsample_bytree=0.7,
                        colsample_bylevel=1,
                        reg_alpha=0,
                        reg_lambda=1,
                        scale_pos_weight=1,
                        seed=1440,
                        missing=None)
xlf.fit(train_x, train_y, eval_metric='error', verbose=True, eval_set=[(valid_x, valid_y)], early_stopping_rounds=30)
# 这个verbose主要是调节系统输出的，如果设置成10,便是每迭代10次就有输出。
# 注意我们这里eval_metric=‘error’便是准确率。这里面并没有accuracy命名的函数，网上大多例子为auc，我这里特意放了个error。
y_pred = xlf.predict(valid_x, ntree_limit=xlf.best_ntree_limit)
auc_score = roc_auc_score(valid_y, y_pred)
y_pred = xlf.predict(valid_x, ntree_limit=xlf.best_ntree_limit)
# xgboost没有直接使用效果最好的树作为模型的机制，这里采用最大树深限制的方法，目的是获取刚刚early_stopping效果最好的，实测性能可以
auc_score = roc_auc_score(valid_y, y_pred)  # 算一下预测结果的roc值