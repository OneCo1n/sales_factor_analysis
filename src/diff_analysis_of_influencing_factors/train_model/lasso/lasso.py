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


def use_lasso(x_train, x_test, y_train, y_test, target_name):

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



def use_lassoLARS(x_train, x_test, y_train, y_test, target_name):
    # 构造不同的lambda值
    Lambdas = np.logspace(-5, 2, 200)

    # 设置交叉验证的参数，使用均方误差评估
    lassoLARS_cv = LassoLarsCV(normalize=False, cv=10,n_jobs=3, max_iter=10000)
    lassoLARS_cv.fit(x_train, y_train)

    # 基于最佳lambda值建模
    lassoLARS = LassoLars(alpha=.1)
    # lassoLARS = LassoLars(alpha=lassoLARS_cv.alpha_)
    lassoLARS.fit(x_train, y_train)

    # 模型评估
    # lasso_pred = lasso.predict(x_test)
    lassoLARS_pred = lassoLARS.predict(x_test)

    # 均方误差
    mse = mean_squared_error(y_test, lassoLARS_pred)
    rmse = sqrt(mse)
    mae = mean_absolute_error(y_test, lassoLARS_pred)
    r2 = r2_score(y_test, lassoLARS_pred)

    print("[Lasso-LARS] mse: ", mse)
    print("[Lasso-LARS] rmse: ", rmse)
    print("[Lasso-LARS] mae: ", mae)
    print("[Lasso-LARS] r2: ", r2)

    coef_col = (lassoLARS.coef_)

    coef_col_list = list(coef_col)
    features_list = list(x_train.columns)

    features_coef_dict = dict()

    for i in range(len(coef_col_list)):
        features_coef_dict[features_list[i]] = coef_col_list[i]
    print(features_coef_dict)


    selected_features = (x_train.columns[(coef_col != 0)])
    # print(selected_features)

    rejected_features = x_train.columns[(coef_col == 0)]
    print("[Lasso-LARS] rejected_features: ")
    print(rejected_features)

    # 返回系数为0的特征
    return list(rejected_features)