from diff_analysis_of_influencing_factors.data_set.data_division import model_data_division
from diff_analysis_of_influencing_factors.train_model.lightGBM.lightGBM import use_lightGBM
import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn.linear_model import Lasso, LassoCV, LassoLars,LassoLarsCV
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.kernel_ridge import KernelRidge
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from math import sqrt
from sklearn.metrics import r2_score  # R square
import asgl


def do_analysis_use_lasso_lightGBM(data, target_name):
    # 特征选择 数据集划分

    x_train, x_test, y_train, y_test = model_data_division(data, target_name, 0.2)
    # 特征筛选模块 使用lasso
    rejected_features = feature_screening_use_lasso(x_train, x_test, y_train, y_test, target_name)
    #rejected_features = use_lassoLARS(x_train, x_test, y_train, y_test, target_name)
    #rejected_features = use_adaptive_kernel_lasso(x_train, x_test, y_train, y_test, target_name)
    # 完成特征筛选
    selected_data = features_select_from_rejected_list(data, rejected_features)
    # 对特征筛选后的数据集进行划分
    X_train, X_test, Y_train, Y_test = model_data_division(selected_data, target_name, 0.2)

    # 信息增益计算模块 使用lightGBM 数据传入模型开始训练  获取分析结果
    mse, rmse, r2, mae, mape, importance_dict = use_lightGBM(X_train, X_test, Y_train, Y_test, target_name)
    return mse, rmse, r2, mae, mape, importance_dict


def feature_screening_use_lasso(x_train, x_test, y_train, y_test, target_name):

    # 构造不同的lambda值
    Lambdas = np.logspace(-5, 2, 200)

    # 设置交叉验证的参数，使用均方误差评估
    # 核函数为rbf
    KernelRidge(alpha=0.6, kernel='rbf', degree=2, coef0=2.5)
    # 自适应Lasso
    lasso_cv = LassoCV(alphas=Lambdas, normalize=True, cv=10, max_iter=10000)
    lasso_cv.fit(x_train, y_train)

    # 基于最佳lambda值建模
    lasso = Lasso(alpha=lasso_cv.alpha_, normalize=True, max_iter=10000)
    lasso.fit(x_train, y_train)

    # 模型评估
    lasso_pred = lasso.predict(x_test)

    # 均方误差
    mse = mean_squared_error(y_test, lasso_pred)
    rmse = sqrt(mse)
    mae = mean_absolute_error(y_test, lasso_pred)
    r2 = r2_score(y_test, lasso_pred)

    print("[Lasso] mse: ", mse)
    print("[Lasso] rmse: ", rmse)
    print("[Lasso] mae: ", mae)
    print("[Lasso] r2: ", r2)

    coef_col = (lasso.coef_)

    coef_col_list = list(coef_col)
    features_list = list(x_train.columns)

    features_coef_dict =  dict()

    for i in range(len(coef_col_list)):
        features_coef_dict[features_list[i]] = coef_col_list[i]
    print(features_coef_dict)

    selected_features =(x_train.columns[(coef_col != 0)])
    # print(selected_features)

    rejected_features =  x_train.columns[(coef_col == 0)]
    print("[Lasso] rejected_features: ")
    print(rejected_features)

    # 返回系数为0的特征
    return list(rejected_features)

def features_select_from_rejected_list(data, rejected_features):

    select_features = data.drop(labels=None, axis=1, index=None, columns=rejected_features, inplace=False)

    return select_features