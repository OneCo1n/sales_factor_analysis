## pca特征降维
# 导入相关模块
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from numpy.linalg import eig
from sklearn.datasets import load_iris
import csv
from compnents.material_plant_pearson import *
from matplotlib import pyplot
from matplotlib.font_manager import FontProperties

font_set = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=15)

# 000000000070251989
# # 导入数据
# print("正在下载数据集......")
# iris = load_iris() # 150*4的矩阵，4个特征，行是个数，列是特征
# print(iris)
# X = iris.data
# print(X.dtype)
material = input('input the material id: ')
df_pearson, dfDsv = oneMaterialOnAllPlantPearson(material)
dfDsv = dfDsv.drop(labels=None, axis=1, index=None, columns=['quantity'], inplace=False)
k = 10  #选取贡献最大的前几个特征作为主成分，根据实际情况灵活选取
# Standardize by remove average通过去除平均值进行标准化
# print("标准化......")
# X = X - X.mean(axis=0)

# Calculate covariance matrix:计算协方差矩阵：
print("计算协方差矩阵......")
X_cov = np.cov(dfDsv.T, ddof=0)

# Calculate  eigenvalues and eigenvectors of covariance matrix
# 计算协方差矩阵的特征值和特征向量
print("计算协方差矩阵的特征值和特征向量......")
eigenvalues, eigenvectors = eig(X_cov)

# top k large eigenvectors选取前k个特征向量
klarge_index = eigenvalues.argsort()[-k:][::-1]
k_eigenvectors = eigenvectors[klarge_index]
print('klarge_index: {}'.format(klarge_index))
# X和k个特征向量进行点乘
X_pca = np.dot(dfDsv, k_eigenvectors.T)

## 绘制贡献率图像
print('eigenvectors: {}' .format(eigenvectors))
# print(eigenvalues)
tot=sum(eigenvalues)   # 计算特征值的和
print('tot : {}' .format(tot))
var_exp=[(i / tot) for i in sorted(eigenvalues,reverse=True)] # 按照降序排列特征值，并计算贡献率
print('var_exp: {}' .format(var_exp) )
cum_var_exp=np.cumsum(var_exp)  #累计贡献度
plt.bar(range(1,len(var_exp)+1),var_exp,alpha=0.5,align='center',label='individual var') # 绘制柱状图，
#plt.bar(range(1,len(var_exp)+1),var_exp,alpha=0.5,align='center',label=u'贡献度', fontproperties=font_set) # 绘制柱状图，

plt.step(range(1,len(var_exp)+1),cum_var_exp,where='mid',label='cumulative var')   # 绘制阶梯图
#plt.step(range(1,len(var_exp)+1),cum_var_exp,where='mid',label='累计贡献度', fontproperties=font_set)   # 绘制阶梯图

plt.xticks()
plt.ylabel(u'重要程度', fontproperties=font_set)       # 纵坐标
plt.xlabel(u'特征', fontproperties=font_set) # 横坐标
plt.legend(loc='best')  # 图例位置，右下角
plt.show()




