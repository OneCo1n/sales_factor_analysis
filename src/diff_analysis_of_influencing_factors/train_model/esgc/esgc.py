from diff_analysis_of_influencing_factors.data_set.data_division import model_data_division
from diff_analysis_of_influencing_factors.train_model.lasso.alasso import use_adaptive_kernel_lasso
from diff_analysis_of_influencing_factors.train_model.lasso.lasso import use_lasso, use_lassoLARS
from diff_analysis_of_influencing_factors.train_model.lightGBM.lightGBM import use_lightGBM
from sklearn.metrics import mean_squared_error  # 均方误差
from sklearn.metrics import mean_absolute_error  # 平方绝对误差
from sklearn.metrics import r2_score  # R square
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.model_selection import GridSearchCV
from matplotlib import pyplot
from xgboost import XGBClassifier
from xgboost import plot_importance
from data_preprocess.columnsExg import *
import lightgbm as lgb
import numpy as np
from sklearn import metrics
from diff_analysis_of_influencing_factors.train_model.qga.qga import QGA
import matplotlib.pyplot as plt
from diff_analysis_of_influencing_factors.data_set.data_division import model_data_division


def do_analysis_use_esgc(data, target_name):
    # 特征选择 数据集划分

    x_train, x_test, y_train, y_test = model_data_division(data, target_name, 0.2)
    # 特征值筛选模块 AKL
    rejected_features = use_adaptive_kernel_lasso(x_train, x_test, y_train, y_test, target_name)
    # 完成特征筛选
    selected_data = features_select_from_rejected_list(data, rejected_features)
    # 对特征筛选后的数据集进行划分
    X_train, X_test, Y_train, Y_test = model_data_division(selected_data, target_name, 0.2)
    # 数据传入模型开始训练  QGA-lightGBM 获取分析结果
    mse, rmse, r2, mae, mape, importance_dict = use_qga_lightGBM(X_train, X_test, Y_train, Y_test, target_name)
    return mse, rmse, r2, mae, mape, importance_dict


def features_select_from_rejected_list(data, rejected_features):

    select_features = data.drop(labels=None, axis=1, index=None, columns=rejected_features, inplace=False)

    return select_features



def use_qga_lightGBM(x_train, x_test, y_train, y_test, target_name):


    # 创建成lgb特征的数据集格式
    lgb_train = lgb.Dataset(x_train, y_train)  # 将数据保存到LightGBM二进制文件将使加载更快
    lgb_eval = lgb.Dataset(x_test, y_test, reference=lgb_train)  # 创建验证数据

    # 将参数写成字典下形式
    params = {
        'task': 'train',
        'boosting_type': 'dart',  # 设置提升类型
        'objective': 'regression',  # 目标函数

        'metric': {'l2', 'auc'},  # 评估函数
        'num_leaves': 3000,  # 叶子节点数
        'learning_rate': 0.1,  # 学习速率
        'feature_fraction': 0.9,  # 建树的特征选择比例
        'bagging_fraction': 0.8,  # 建树的样本采样比例
        'bagging_freq': 5,  # k 意味着每 k 次迭代执行bagging
        'verbose': 1  # <0 显示致命的, =0 显示错误 (警告), >0 显示信息
    }

    print('[info : lightGBM] 开始训练模型')
    # 训练 cv and train
    gbm = lgb.train(params, lgb_train, num_boost_round=30, valid_sets=lgb_eval,
                    early_stopping_rounds=100)  # 训练数据需要参数列表和数据集

    print('[info : lightGBM] 开始验证')
    # 预测数据集
    y_pred = gbm.predict(x_test, num_iteration=gbm.best_iteration)  # 如果在训练期间启用了早期停止，可以通过best_iteration方式从最佳迭代中获得预测
    # 评估模型

    mse = mean_squared_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    r2 = r2_score(y_test, y_pred)
    mae = metrics.mean_absolute_error(y_test, y_pred)
    mape = calculate_mape(y_test , y_pred )
    # mape = cal_mape(y_test, y_pred)
    print('[info : LightGBM] The mse of prediction is:', mse)
    print('[info : LightGBM] The rmse of prediction is:', rmse)  # 计算真实值和预测值之间的均方根误差
    print('[info : LightGBM] the r2 of prediction is:', r2)
    print('[info : LightGBM] the mae of prediction is:', mae)
    print('[info : LightGBM] the mape of prediction is:', mape)

    importance = gbm.feature_importance()
    feature_name = gbm.feature_name()
    importance_dict = dict()

    sum = 0
    count = 0
    for i, label in enumerate(feature_name):
        sum += float(importance[i])
        importance_dict[label] = float(importance[i])

    for i, label in enumerate(feature_name):
        importance_dict[label] = float(importance[i] / sum)

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
    # pyplot.bar(range(len(gbm.feature_importance())), gbm.feature_importance())
    # plot_importance(importance_dict)
    # pyplot.show()

    return mse, rmse, r2, mae, mape, importance_dict

def calculate_mape(y_true, y_pred):
    return np.mean(np.abs((y_pred - y_true) / (y_true) ))


