import xgboost as xgb
import numpy as np
import pickle
from sklearn.model_selection import KFold, train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, mean_squared_error #混淆矩阵，均方误差
from sklearn.datasets import load_iris, load_digits, load_boston
#鸢尾花数据集，鸢尾花种类预测，属于分类，分类问题用混淆矩阵
#样本数据集,属于分类，分类问题用混淆矩阵
#波士顿房价数据集，线性回归，回归问题用MSE

rng = np.random.RandomState(31337)

#回归问题：MSE
print("\n波士顿房价回归预测问题")
boston = load_boston()
y = boston['target']
X = boston['data']
kf = KFold(n_splits=2, shuffle=True, random_state=rng)
print("在2折数据上的交叉验证")
for train_index, test_index in kf.split(X):
    xgb_model = xgb.XGBRegressor().fit(X[train_index],y[train_index])
    predictions = xgb_model.predict(X[test_index])
    actuals = y[test_index]
    print("MSE:",mean_squared_error(actuals, predictions))
'''
波士顿房价回归预测问题
在2折数据上的交叉验证
MSE: 21.88594990885867
MSE: 14.807648754688827
'''

# 第2种训练方法的 调参方法：使用sklearn接口的regressor + GridSearchCV
print("参数最优化：")
y = boston['target']
X = boston['data']
xgb_model = xgb.XGBRegressor()
param_dict = {'max_depth': [2, 4, 6],
              'n_estimators': [50, 100, 200]}

clf = GridSearchCV(xgb_model, param_dict, verbose=1)
'''
verbose是控制日志输出的, 'controls the verbosity: the higher, the more messages'
verbose = 0没有输出；verbose = 1 简化版日志输出；verbose=2 更细致的日志输出...
一般设置到2就很多信息了，日志输出太多会影响运行的速度。
'''
clf.fit(X, y)
print(clf.best_score_)
print(clf.best_params_)
'''
参数最优化：
Fitting 5 folds for each of 9 candidates, totalling 45 fits
0.6839859272017424
{'max_depth': 2, 'n_estimators': 100}
'''
