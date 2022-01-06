from data_preprocess.missing_value_processing import missing_value_fill_num_0
from db import data_extraction
from sklearn.model_selection import train_test_split
from data_encoding.encoder import *
from correlation_analysis.pearson import *
from db.data_extraction import getPlantJoinNumofpeopleCode
from db.data_storage import insert_table_pearson
from factor_analysis.factor_analysis import *
from db.dataQuery import *
from data_preprocess.standardizing import *
from data_preprocess.outlierProcessing import *
from plot.pairPlot import *
from db.data_storage import *
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.metrics import r2_score
#
# #def main():
# # 产生一些稀疏数据
# np.random.seed(42)
# n_samples, n_features = 50, 200
# X = np.random.randn(n_samples, n_features) # randn(...)产生的是正态分布的数据
# coef = 3 * np.random.randn(n_features)     # 每个特征对应一个系数
# inds = np.arange(n_features)
# np.random.shuffle(inds)
# coef[inds[10:]] = 0  # 稀疏化系数--随机的把系数向量1x200的其中10个值变为0
# y = np.dot(X, coef)  # 线性运算 -- y = X.*w
# # 添加噪声：零均值，标准差为 0.01 的高斯噪声
# y += 0.01 * np.random.normal(size=n_samples)
#
# # 把数据划分成训练集和测试集
# n_samples = X.shape[0]
# X_train, y_train = X[:n_samples // 2], y[:n_samples // 2]
# X_test, y_test = X[n_samples // 2:], y[n_samples // 2:]
#
# # 训练 Lasso 模型
# from sklearn.linear_model import Lasso
# alpha = 0.1
# lasso = Lasso(alpha=alpha)
# y_pred_lasso = lasso.fit(X_train, y_train).predict(X_test)
# r2_score_lasso = r2_score(y_test, y_pred_lasso)
# print(lasso)
# print("r^2 on test data : %f" % r2_score_lasso)
#
# # 训练 ElasticNet 模型
# from sklearn.linear_model import ElasticNet
# enet = ElasticNet(alpha=alpha, l1_ratio=0.7)
# y_pred_enet = enet.fit(X_train, y_train).predict(X_test)
# r2_score_enet = r2_score(y_test, y_pred_enet)
# print(enet)
# print("r^2 on test data : %f" % r2_score_enet)
#
# plt.plot(enet.coef_, color='lightgreen', linewidth=2,
#          label='Elastic net coefficients')
# plt.plot(lasso.coef_, color='gold', linewidth=2,
#          label='Lasso coefficients')
# plt.plot(coef, '--', color='navy', label='original coefficients')
# plt.legend(loc='best')
# plt.title("Lasso R^2: %f, Elastic Net R^2: %f"
#           % (r2_score_lasso, r2_score_enet))
# plt.show()
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 17:07:16 2018
@author: wp:lasso|ridge
"""
# 经典鸢尾花数据集
from sklearn.datasets import load_iris
material = '000000000070309748'
df = data_extraction.get_df_from_db(material)
iris = load_iris()

data_x = iris.data
print("data_x-------------")
print(data_x)

data_y = iris.target
print("data_y-------------")
new_adv_data = df.iloc[:, 0:]
print(data_y)

X_train, X_test, Y_train, Y_test = train_test_split(new_adv_data.iloc[:, 1:34], new_adv_data.quantity,
                                                        test_size=0.25,train_size=0.75)



# 带入需要的包、库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.linear_model import Lasso, LassoCV
from sklearn.metrics import mean_squared_error

x_tr, x_te, y_tr, y_te = train_test_split(data_x, data_y, train_size=0.8, random_state=22)
######################ridge########################################
# 通过不同的alpha值 生成不同的ridge模型
alphas = 10 ** np.linspace(-10, 10, 100)
ridge_cofficients = []

for alpha in alphas:
    ridge = Ridge(alpha=alpha, normalize=True)
    ridge.fit(x_tr, y_tr)
    ridge_cofficients.append(ridge.coef_)

# 画出alpha和回归系数的关系
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
# 设置绘图风格
plt.style.use('ggplot')
plt.plot(alphas, ridge_cofficients)
plt.xscale('log')
plt.axis('tight')
plt.title(r'alpha系数与岭回归系数的关系')
plt.xlabel('Log Alpha')
plt.ylabel('Cofficients')
plt.show()

# ridge交叉验证
ridge_cv = RidgeCV(alphas=alphas, normalize=True, cv=10)
ridge_cv.fit(x_tr, y_tr)
# 取出最佳的lambda值ridge_best_alpha = ridge_cv.alpha_
ridge_best_alpha = ridge_cv.alpha_  # 得到最佳lambda值
# 基于最佳lambda值建模
ridge = Ridge(alpha=ridge_best_alpha, normalize=True)
ridge.fit(x_tr, y_tr)
ridge_predict = ridge.predict(x_te)
rmse = np.sqrt(mean_squared_error(y_te, ridge_predict))

######################lasso##################################


# LASSO回归模型的交叉验证
lasso_cv = LassoCV(alphas=alphas, normalize=True, cv=10, max_iter=10000)
lasso_cv.fit(x_tr, y_tr)
# 取出最佳的lambda值
lasso_best_alpha = lasso_cv.alpha_
lasso_best_alpha
# 基于最佳lambda值建模
lasso = Lasso(alpha=lasso_best_alpha, normalize=True, max_iter=10000)
lasso.fit(x_tr, y_tr)

lasso_predict = lasso.predict(x_te)  # 预测

RMSE = np.sqrt(mean_squared_error(y_te, lasso_predict))
print(RMSE)

from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
x = scaler.fit_transform(iris.data)
y = iris.target
lasso = Lasso(alpha=0.2)
lasso.fit(x, y)
lasso.coef_