# coding: utf-8
# pylint: disable = invalid-name, C0111

import matplotlib.pylab as plt
from sklearn.metrics import mean_squared_error #均方误差
from sklearn.metrics import mean_absolute_error #平方绝对误差
from sklearn.metrics import r2_score#R square
import json
import lightgbm as lgb
import pandas as pd
from sklearn import model_selection
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

from compnents.material_plant_pearson import oneMaterialOnAllPlantPearson

#iris = load_iris()  # 载入鸢尾花数据集
#data = iris.data
#target = iris.target
# X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2)
#000000000070251989
#000000000070309748
material = '000000000070251989'
#data = data_extraction.get_df_from_db(material)
pearson, data = oneMaterialOnAllPlantPearson(material)
print(data)
#data=pd.read_excel(r'C:\Users\Administrator\Desktop\diabetes.xlsx')
#predictors=data.columns[:-1]
predictors = data['quantity']
data = data.iloc[:, 0:]
target = data.quantity
#data=data.drop(['AGE','SEX'],axis=1)
#拆分为训练集和测试集

features = data.iloc[:, 1:34]

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2)
# x_train,x_test,y_train,y_test=model_selection.train_test_split(data.iloc[:, 1:34], data.quantity,
#                                                         test_size=0.25,train_size=0.75)

# 加载你的数据
# print('Load data...')
# df_train = pd.read_csv('../regression/regression.train', header=None, sep='\t')
# df_test = pd.read_csv('../regression/regression.test', header=None, sep='\t')
#
# y_train = df_train[0].values
# y_test = df_test[0].values
# X_train = df_train.drop(0, axis=1).values
# X_test = df_test.drop(0, axis=1).values

# 创建成lgb特征的数据集格式
lgb_train = lgb.Dataset(X_train, y_train)  # 将数据保存到LightGBM二进制文件将使加载更快
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)  # 创建验证数据

# 将参数写成字典下形式
params = {
    'task': 'train',
    'boosting_type': 'gbdt',  # 设置提升类型
    'objective': 'regression',  # 目标函数
    'metric': {'l2', 'auc'},  # 评估函数
    'num_leaves': 3000,  # 叶子节点数
    'learning_rate': 0.1,  # 学习速率
    'feature_fraction': 0.9,  # 建树的特征选择比例
    'bagging_fraction': 0.8,  # 建树的样本采样比例
    'bagging_freq': 5,  # k 意味着每 k 次迭代执行bagging
    'verbose': 1  # <0 显示致命的, =0 显示错误 (警告), >0 显示信息
}

print('Start training...')
# 训练 cv and train
gbm = lgb.train(params, lgb_train, num_boost_round=30, valid_sets=lgb_eval, early_stopping_rounds=100)  # 训练数据需要参数列表和数据集

print('Save model...')

gbm.save_model('model.txt')  # 训练后保存模型到文件

print('Start predicting...')
# 预测数据集
y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)  # 如果在训练期间启用了早期停止，可以通过best_iteration方式从最佳迭代中获得预测
# 评估模型
print('The mse of prediction is:', mean_squared_error(y_test, y_pred))
print('The rmse of prediction is:', mean_squared_error(y_test, y_pred) ** 0.5)  # 计算真实值和预测值之间的均方根误差
print('the r2 of prediction is:',r2_score(y_test, y_pred))
#print('Feature importances:', list(gbm.get_fscore()))
plt.figure(figsize=(12,6))
lgb.plot_importance(gbm)
plt.title("Featurertances")
plt.show()

from sklearn.model_selection import GridSearchCV
estimator = lgb.LGBMRegressor(num_leaves=31)
param_grid = {
    'learning_rate': [0.01, 0.1, 1],
    'n_estimators': [20, 40]
}
gbm = GridSearchCV(estimator, param_grid)
gbm.fit(X_train, y_train)
print('Best parameters found by grid search are:', gbm.best_params_)