from diff_analysis_of_influencing_factors.data_set.data_division import model_data_division
from sklearn import model_selection
from sklearn.linear_model import Lasso, LassoCV, LassoLars, LassoLarsCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.kernel_ridge import KernelRidge
import numpy as np
from math import sqrt
from sklearn.metrics import r2_score  # R square
import lightgbm as lgb
from sklearn import metrics

def do_analysis_use_AKL_PSO_lightGBM(data, target_name):
    # 特征提取 数据集划分
    x_train, x_test, y_train, y_test = model_data_division(data, target_name, 0.2)
    # 特征筛选模块 使用 adaptive kernel lasso(AKL)
    rejected_features = feature_screening_use_AKL(x_train, x_test, y_train, y_test, target_name)
    # 完成特征筛选
    selected_data = features_select_from_rejected_list(data, rejected_features)
    # 对特征筛选后的数据集进行划分
    X_train, X_test, Y_train, Y_test = model_data_division(selected_data, target_name, 0.2)
    # 信息增益计算模块 使用PSO-lightGBM 数据传入模型开始训练  获取分析结果
    mse, rmse, r2, mae, mape, importance_dict = use_PSO_lightGBM(X_train, X_test, Y_train, Y_test, target_name)
    return mse, rmse, r2, mae, mape, importance_dict


def feature_screening_use_AKL(x_train, x_test, y_train, y_test, target_name):
    # 构造不同的lambda值
    Lambdas = np.logspace(-5, 2, 200)

    # 设置交叉验证的参数，使用均方误差评估

    # # 自适应核Lasso   核函数为rbf
    KernelRidge(alpha=0.6, kernel='rbf', degree=2, coef0=2.5)
    lasso_cv = LassoCV(alphas=Lambdas, normalize=True, cv=10, max_iter=10000)
    lasso_cv.fit(x_train, y_train)

    # 基于最佳lambda值建模
    AKL = Lasso(alpha=lasso_cv.alpha_, normalize=True, max_iter=10000)
    AKL.fit(x_train, y_train)

    # 模型评估
    lasso_pred = AKL.predict(x_test)

    # 均方误差
    mse = mean_squared_error(y_test, lasso_pred)
    rmse = sqrt(mse)
    mae = mean_absolute_error(y_test, lasso_pred)
    r2 = r2_score(y_test, lasso_pred)

    print("[AKL] mse: ", mse)
    print("[AKL] rmse: ", rmse)
    print("[AKL] mae: ", mae)
    print("[AKL] r2: ", r2)

    coef_col = (AKL.coef_)

    coef_col_list = list(coef_col)
    features_list = list(x_train.columns)

    features_coef_dict = dict()

    for i in range(len(coef_col_list)):
        features_coef_dict[features_list[i]] = coef_col_list[i]
    print(features_coef_dict)

    selected_features = (x_train.columns[(coef_col != 0)])
    # print(selected_features)

    rejected_features = x_train.columns[(coef_col == 0)]
    print("[AKL] rejected_features: ")
    print(rejected_features)

    # 返回系数为0的特征
    return list(rejected_features)


def features_select_from_rejected_list(data, rejected_features):
    select_features = data.drop(labels=None, axis=1, index=None, columns=rejected_features, inplace=False)

    return select_features


def use_PSO_lightGBM(x_train, x_test, y_train, y_test, target_name):

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

    print('[info : PSO_lightGBM] 开始训练模型')
    # 训练 cv and train
    gbm = lgb.train(params, lgb_train, num_boost_round=30, valid_sets=lgb_eval,
                    early_stopping_rounds=100)  # 训练数据需要参数列表和数据集

    print('[info : PSO_lightGBM] 开始验证')
    # 预测数据集
    y_pred = gbm.predict(x_test, num_iteration=gbm.best_iteration)  # 如果在训练期间启用了早期停止，可以通过best_iteration方式从最佳迭代中获得预测
    # 评估模型

    mse = mean_squared_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    r2 = r2_score(y_test, y_pred)
    mae = metrics.mean_absolute_error(y_test, y_pred)
    mape = calculate_mape(y_test , y_pred )
    # mape = cal_mape(y_test, y_pred)
    print('[info : PSO_LightGBM] The mse of prediction is:', mse)
    print('[info : PSO_LightGBM] The rmse of prediction is:', rmse)  # 计算真实值和预测值之间的均方根误差
    print('[info : PSO_LightGBM] the r2 of prediction is:', r2)
    print('[info : PSO_LightGBM] the mae of prediction is:', mae)
    print('[info : PSO_LightGBM] the mape of prediction is:', mape)

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


