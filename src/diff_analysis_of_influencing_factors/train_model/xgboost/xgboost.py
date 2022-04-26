from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_error  # 均方误差
from sklearn.metrics import mean_absolute_error  # 平方绝对误差
from sklearn.metrics import r2_score  # R square
from sklearn.datasets import load_iris
import xgboost as xgb
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import explained_variance_score
import numpy as np
from sklearn import metrics

from diff_analysis_of_influencing_factors.data_set.data_division import model_data_division


def do_analysis_use_xgboost(data, target_name):

    x_train, x_test, y_train, y_test = model_data_division(data, target_name, 0.2)
    # 用于模型训练，得出特征重要度
    mse, rmse, r2, mae, mape, importance_dict = use_xgboost(x_train, x_test, y_train, y_test, target_name)
    return mse, rmse, r2, mae, mape, importance_dict


def use_xgboost(x_train, x_test, y_train, y_test, target_name):
    # 训练模型
    # silent=True,
    model = xgb.XGBClassifier(max_depth=5, learning_rate=0.1, n_estimators=1000,  objective='reg:squarederror')
    model.fit(x_train, y_train)

    # 对测试集进行预测
    y_pred = model.predict(x_test)

    # 计算准确率
    accuracy = explained_variance_score(y_test, y_pred)
    print('accuracy:%2.f%%' % (accuracy * 100))

    # 评估模型
    mse = mean_squared_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    r2 = r2_score(y_test, y_pred)
    mae = metrics.mean_absolute_error(y_test, y_pred)
    mape = calculate_mape(y_test, y_pred)
    print('[info : XGBoost] The mse of prediction is:', mse)
    print('[info : XGBoost] The rmse of prediction is:', rmse)  # 计算真实值和预测值之间的均方根误差
    print('[info : XGBoost] the r2 of prediction is:', r2)
    print('[info : XGBoost] the mae of prediction is:', mae)
    print('[info : XGBoost] the mape of prediction is:', mape)



    importance = model.feature_importances_
    feature_name = model._Booster.feature_names
    importance_dict = dict()

    for i, label in enumerate(feature_name):
        importance_dict[label] = float(importance[i])

    importance_dict = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)

    importance_dict = dict(importance_dict)

    # estimator = lgb.LGBMRegressor(num_leaves=31)
    # param_grid = {
    #     'learning_rate': [0.01, 0.1, 1],
    #     'n_estimators': [20, 40]
    # }
    # gbm = GridSearchCV(estimator, param_grid)
    # gbm.fit(X_train, y_train)
    # print('[info : lightGBM] Best parameters found by grid search are:', gbm.best_params_)
    return mse, rmse, r2, mae, mape, importance_dict

# MAPE需要自己实现
def calculate_mape(y_true, y_pred):
    return np.mean(np.abs((y_pred - y_true) / y_true))
