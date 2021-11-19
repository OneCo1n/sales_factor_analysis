import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib
import datetime
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV
sns.set(style='whitegrid',palette='tab10')
matplotlib.rcParams['font.sans-serif'] = ['SimHei']


#                                 数据读取、展示、缺失值处理

# 数据展示
train=pd.read_csv('F:/论文/Shared-bicycle-usage-forecast-master/train.csv',encoding='utf-8')

print(train.info())

test=pd.read_csv('F:/论文/Shared-bicycle-usage-forecast-master/test.csv',encoding='utf-8')
print(test.info())


#                               通过查看训练集与测试集的详细信息发现都没有缺失值
#                               数据处理
#                               异常值处理
# 描述性统计、异常值检测
print(train.describe().T)

#                               可以看出租赁额（count）数值差异大，再观察一下它们的密度分布：


# sns.set(); np.random.seed(0)
# x = np.random.randn(100)
# ax = sns.distplot(x)
# plt.show()


#                                   观察租赁额密度分布

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
fig.set_size_inches(6,5)
ax = sns.distplot(train['count'])  # will be removed in a future version
#ax = sns.histplot(train['count'])
# sns.displot(train['count'])  # 这种写法 子图和画布会分离
ax.set(xlabel='count',title='租用数量密度分布')
plt.show()


#                                                   离群值处理
# 判断count是否为离群值（异常值），如果不是则不选取（删除）
train_WithoutOutliers = train[np.abs(train['count']-train['count'].mean())<=(3*train['count'].std())]
print(train_WithoutOutliers.shape)
print(train_WithoutOutliers['count'].describe())


fig=plt.figure(figsize=(12,5))
ax1=fig.add_subplot(1,2,1)
ax2=fig.add_subplot(1,2,2)

sns.distplot(train_WithoutOutliers['count'],ax=ax1)
sns.distplot(train['count'],ax=ax2)
ax2.set(xlabel='count',title='去除长尾后count的分布')
ax1.set(xlabel='registered',title='count的分布')
plt.show()
# 数据波动依然很大，而我们希望波动相对稳定，否则容易产生过拟合，所以希望对数据进行处理，使得数据相对稳定，此处选择对数变化，来使得数据稳定。

#                                      对数变换
yLabels=train_WithoutOutliers['count']
yLabels_log=np.log(yLabels)
sns.distplot(yLabels_log)
plt.show();

# 经过对数变换后数据分布更均匀，大小差异也缩小了，使用这样的标签对训练模型是有效果的。接下来对其余的数值型数据进行处理，由于其他数据同时包含在两个数据集中，为方便数据处理先将两个数据集合并。

#
# test['casual']=-1
# test['registered']=-1
# test['count']=-1
#
# Bike_data=pd.concat([train_WithoutOutliers,test],ignore_index=True)
# Bike_data.shape
#
# fig,axes=plt.subplots(2,2)
# fig.set_size_inches(12,10)
#
# sns.distplot(Bike_data['temp'],ax=axes[0,0])
# sns.distplot(Bike_data['atemp'],ax=axes[0,1])
# sns.distplot(Bike_data['humidity'],ax=axes[1,0])
# sns.distplot(Bike_data['windspeed'],ax=axes[1,1])
#
# axes[0,0].set(xlabel='temp',title='温度分布情况')
# axes[0,1].set(xlabel='atemp',title='体感温度分布情况')
# axes[1,0].set(xlabel='humidity',title='湿度分布情况')
# axes[1,1].set(xlabel='windspeed',title='风速分布情况')
#
# #风速不为0的数据的风速的统计描述
# Bike_data[Bike_data['windspeed']!=0]['windspeed'].describe()
