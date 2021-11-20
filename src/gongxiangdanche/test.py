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
matplotlib.rcParams['font.family'] = 'SimHei'
matplotlib.rcParams['font.size'] = 10
matplotlib.rcParams['axes.unicode_minus']=False
#                                 数据读取、展示、缺失值处理

# 数据展示
train=pd.read_csv('E:/WorkSpace/PythonProjects/analysis/train.csv',encoding='utf-8')

print(train.info())

test=pd.read_csv('E:/WorkSpace/PythonProjects/analysis/test.csv',encoding='utf-8')
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
plt.show()

# 经过对数变换后数据分布更均匀，大小差异也缩小了，使用这样的标签对训练模型是有效果的。接下来对其余的数值型数据进行处理，由于其他数据同时包含在两个数据集中，为方便数据处理先将两个数据集合并。


# 为test中加入新列casual、registered、count 值都为 -1
test['casual']=-1
test['registered']=-1
test['count']=-1

# print(test['casual'])
# print(test['registered'])
# print(test['count'])

# 将处理过的train数据与增加新列后的test数据进行合并, test拼接到了train的下面
Bike_data=pd.concat([train_WithoutOutliers,test],ignore_index=True)
Bike_data.shape
# print(Bike_data)

# 设置一个有2*2子图的 画布
fig,axes=plt.subplots(2,2)
fig.set_size_inches(12,10)

# 密度分布图
sns.distplot(Bike_data['temp'],ax=axes[0,0])
sns.distplot(Bike_data['atemp'],ax=axes[0,1])
sns.distplot(Bike_data['humidity'],ax=axes[1,0])
sns.distplot(Bike_data['windspeed'],ax=axes[1,1])

axes[0,0].set(xlabel='temp',title='温度分布情况')
axes[0,1].set(xlabel='atemp',title='体感温度分布情况')
axes[1,0].set(xlabel='humidity',title='湿度分布情况')
axes[1,1].set(xlabel='windspeed',title='风速分布情况')

plt.show()

# 通过观察分布，发现风速为0的数据很多，但通过前边的观察统计可知整体数据是不包含缺失值的，
# 所以可以推测到，数据本身应该是有缺失的，但是用0填充了，这些风速为0的数据会对预测产生干扰，
# 可以使用随机森林根据相同的年份，月份，季节，温度，湿度等几个特征填充下风速的缺失值。

#风速不为0的数据的风速的统计描述
print(Bike_data[Bike_data['windspeed']!=0]['windspeed'].describe())


from sklearn.ensemble import RandomForestRegressor

Bike_data["windspeed_rfr"]=Bike_data["windspeed"]
# 将数据分成风速等于0和不等于两部分
dataWind0 = Bike_data[Bike_data["windspeed_rfr"]==0]
dataWind0 = dataWind0.copy() # 不加这句会报错
dataWind0.loc[:,'year']=pd.DatetimeIndex(dataWind0['datetime']).year
dataWind0.loc[:,'month']=pd.DatetimeIndex(dataWind0['datetime']).month
dataWindNot0 = Bike_data[Bike_data["windspeed_rfr"]!=0]
dataWindNot0 = dataWindNot0.copy()
dataWindNot0.loc[:,'year']=pd.DatetimeIndex(dataWindNot0['datetime']).year
dataWindNot0.loc[:,'month']=pd.DatetimeIndex(dataWindNot0['datetime']).month
#选定模型
rfModel_wind = RandomForestRegressor(n_estimators=1000,random_state=42)
# 选定特征值
windColumns = ["season","weather","humidity","month","temp","year","atemp"]
# 将风速不等于0的数据作为训练集，fit到RandomForestRegressor之中
rfModel_wind.fit(dataWindNot0[windColumns], dataWindNot0["windspeed_rfr"])
# 通过训练好的模型预测风速
wind0Values = rfModel_wind.predict(X= dataWind0[windColumns])
#将预测的风速填充到风速为零的数据中
dataWind0.loc[:,"windspeed_rfr"] = wind0Values
#连接两部分数据
Bike_data = dataWindNot0.append(dataWind0)
Bike_data.reset_index(inplace=True)
Bike_data.drop('index',inplace=True,axis=1)

fig, axes = plt.subplots(2, 2)
fig.set_size_inches(12,10)

sns.distplot(Bike_data['temp'],ax=axes[0,0])
sns.distplot(Bike_data['atemp'],ax=axes[0,1])
sns.distplot(Bike_data['humidity'],ax=axes[1,0])
sns.distplot(Bike_data['windspeed_rfr'],ax=axes[1,1])

axes[0,0].set(xlabel='temp',title='温度分布情况')
axes[0,1].set(xlabel='atemp',title='体感温度分布情况')
axes[1,0].set(xlabel='humidity',title='湿度分布情况')
axes[1,1].set(xlabel='windseed',title='风速分布情况')

plt.show()

#                                           时间型数处理

Bike_data['date']=pd.DatetimeIndex(Bike_data['datetime']).date
Bike_data['hour']=pd.DatetimeIndex(Bike_data['datetime']).hour
Bike_data['year']=pd.DatetimeIndex(Bike_data['datetime']).year
Bike_data['month']=pd.DatetimeIndex(Bike_data['datetime']).month
Bike_data['weekday']=pd.DatetimeIndex(Bike_data['datetime']).weekday

print(Bike_data)




#                                          数据分析
# 描述性分析
print(Bike_data.describe().T)
# 温度, 体表温度, 相对湿度, 风速均近似对称分布, 而非注册用户, 注册用户,以及总数均右边分布。

for i in range(5, 12):
    name = train.columns[i]
    print('{0}偏态系数为 {1}, 峰态系数为 {2}'.format(name, train[name].skew(), train[name].kurt()))

# temp, atemp, humidity低度偏态,
# windspeed中度偏态,
# casual, registered, count高度偏态；
# temp, atemp, humidity为平峰分布,
# windspeed,casual, registered, count为尖峰分布。

#                                           探索性分析
# 整体性分析

n_Bike_data=Bike_data.loc[(Bike_data['casual']!=-1)&(Bike_data['registered']!=-1)&(Bike_data['count']!=-1)]
print(n_Bike_data)

sns.pairplot(n_Bike_data ,
             height=4,
             x_vars=['holiday','workingday','weather','season',
                                'weekday','hour','windspeed_rfr','humidity','temp','atemp'] ,
             y_vars=['casual','registered','count'] , plot_kws={'alpha': 0.1})
plt.show()
#大致可以看出：
#
# 会员在工作日出行多，节假日出行少，临时用户则相反；
# 一季度出行人数总体偏少；
# 租赁数量随天气等级上升而减少；
# 小时数对租赁情况影响明显，会员呈现两个高峰，非会员呈现一个正态分布；
# 租赁数量随风速增大而减少；
# 温度、湿度对非会员影响比较大，对会员影响较小。


#                                                             相关性分析
#各个特征与每小时租车总量（count）的相关性

correlation = n_Bike_data.corr()
mask = np.array(correlation)
mask[np.tril_indices_from(mask)] = False
fig,ax= plt.subplots()
fig.set_size_inches(20,15)
sns.heatmap(correlation, mask=mask,vmax=.8, square=True,annot=True)

plt.show()

# count 和 registered、casual高度正相关，相关系数分别为0.7 与0.97。
# 因为 count = casual + registered ，所以这个正相关和预期相符。
# count 和 temp 正相关，相关系数为 0.39。一般来说，气温过低人们不愿意骑车出行。
# count 和 humidity（湿度）负相关，湿度过大的天气不适宜骑车。当然考虑湿度的同时也应该考虑温度。
# windspeed似乎对租车人数影响不大（0.1），但我们也应该考虑到极端大风天气出现频率应该不高。
# 风速在正常范围内波动应该对人们租车影响不大。
# 可以看出特征值对租赁数量的影响力度为,时段>温度>湿度>年份>月份>季节>天气等级>风速>星期几>是否工作日>是否假日

#                                                        影响因素分析
# 时段对租赁数量的影响

workingday_df=n_Bike_data[n_Bike_data['workingday']==1]
workingday_df = workingday_df.groupby(['hour'], as_index=True).agg({'casual':'mean',
                                                                    'registered':'mean',
                                                                    'count':'mean'})

nworkingday_df=Bike_data[Bike_data['workingday']==0]
nworkingday_df = nworkingday_df.groupby(['hour'], as_index=True).agg({'casual':'mean',
                                                                      'registered':'mean',
                                                                      'count':'mean'})
fig, axes = plt.subplots(1, 2,sharey = True)

workingday_df.plot(figsize=(15,5),title = '工作日平均每小时的租赁情况',ax=axes[0])
nworkingday_df.plot(figsize=(15,5),title = '休息日平均每小时的租赁情况',ax=axes[1])

plt.show()

# 通过观察上述两个图可以看到 ·工作日：
#
# 对于会员用户来说两个用车高峰为早7 8点左右，晚5 6点左右，这段时间一般是上下班时间，中午的时候也会有个小高峰，可能是中午外出就餐。
# 对于非会员用户来说用户欺负比较平缓，高峰期在17点左右，可能是出去买菜或者吃饭。
# 通过以上观察可以发现，会员用户的用车数量远超过非会员用户 ·非工作日
# 租赁数量随着一天之内时间的变化呈现一个正态分布，用车最多时间在14点左右，最低在4点左右


# 温度对租赁数量的影响
# 数据按小时统计展示起来太麻烦，希望能够按天汇总取一天的气温中位数
temp_df = n_Bike_data.groupby(['date', 'weekday'], as_index=False).agg({'year': 'mean',
                                                                        'month': 'mean',
                                                                        'temp': 'median'})

# 预计按天统计的波动仍然很大，再按月取日平均值
temp_month = temp_df.groupby(['year', 'month'], as_index=False).agg({'weekday': 'min',
                                                                     'temp': 'median'})
# 将按天求和统计数据的日期转换成datetime格式
temp_df['date'] = pd.to_datetime(temp_df['date'])

# 设置画框尺寸
fig = plt.figure(figsize=(18, 6))
ax = fig.add_subplot(1, 1, 1)

# 使用折线图展示总体租赁情况（count）随时间的走势
plt.plot(temp_df['date'], temp_df['temp'], linewidth=1.3, label='Daily average')
ax.set_title('温度走势')

ax.legend()
plt.show()

temp_rentals = n_Bike_data.groupby(['temp'], as_index=True).agg({'casual':'mean',
                                                                       'registered':'mean',
                                                                       'count':'mean'})
temp_rentals.plot(title = '不同温度下租车情况')
plt.show()


# 湿度对租赁数量的影响
humidity_rentals = n_Bike_data.groupby(['humidity'], as_index=True).agg({'casual':'mean',
                                                                       'registered':'mean',
                                                                       'count':'mean'})
humidity_rentals.plot(title = '不同湿度下租车情况')
plt.show()

# 年份、月份对租赁数量的影响
year_rentals = n_Bike_data.groupby(['year'], as_index=True).agg({'casual':'mean',
                                                                       'registered':'mean',
                                                                       'count':'mean'})
year_rentals.plot(title = '每年租车情况')
plt.show()

month_rentals = n_Bike_data.groupby(['year','month'], as_index=True).agg({'casual':'mean',
                                                                       'registered':'mean',
                                                                       'count':'mean'})
month_rentals.plot(title = '每月租车情况')
plt.show()

# 通过观察以上两个图可以发现：
# 共享单车整体租赁情况2011到2012是有所增长的
# 在每年1月份租车数量达到全年最低，7 8月达到全年最高
# 上半年租车数量逐渐提高，下半年租车数量逐渐降低
# 通过前边的温度变化情况可推测出来，上半年租车数量逐渐提高，下半年租车数量逐渐降低可能是因为温度的原因，
# 上半年温度逐渐升高，华盛顿不会感觉那么冷，适合骑单车，下半年温度逐渐降低，不适合骑单车，所以呈现如图所示变化趋势
# 12-次年一月份租车数量最低，推测一方面是温度原因不适合骑单车，另一方面通过前边的观察可知，大部分使用共享单车的人都是上班人士，
# 因为圣诞节美国大部分工作人员放假，所以对单车的使用量也减少


# 季节对租赁数量的影响

day_df=n_Bike_data.groupby('date').agg({'year':'mean','season':'mean',
                                      'casual':'sum', 'registered':'sum'
                                      ,'count':'sum','temp':'mean',
                                      'atemp':'mean'})

season_df = day_df.groupby(['year','season'], as_index=True).agg({'casual':'mean',
                                                                  'registered':'mean',
                                                                  'count':'mean'})

season_df.plot(figsize=(18,6),title = '随着季节变化每天使用量变化趋势')
plt.show()

temp_df = day_df.groupby(['year','season'], as_index=True).agg({'temp':'mean',
                                                                'atemp':'mean'})
temp_df.plot(figsize=(18,6),title = '随着季节变化每天温度变化趋势')
plt.show()

# 可以看出无论是临时用户还是会员用户用车的数量都在秋季迎来高峰，而春季度用户数量最低。
#
# 天气情况对租赁数量的影响
# 不同天气的天数不同，例如非常糟糕的天气（4）会很少出现，查看一下不同天气等级的数据条数，再对租赁数量按天气等级取每小时平均值。

count_weather = n_Bike_data.groupby('weather')
print(count_weather[['casual','registered','count']].count())

weather_df = n_Bike_data.groupby('weather', as_index=True).agg({'casual':'mean',
                                                              'registered':'mean'})
weather_df.plot.bar(stacked=True,title = '不同天气下每个小时的平均租赁数量')
plt.show()

#天气等级4的时候出行人数并不少，尤其是会员出行人数甚至比天气等级2的平均值还高，按理说4等级的应该是最少的，将天气等级4的数据打印出来找一下原因：
n_Bike_data[Bike_data['weather']==4]
# 观察可知该数据是在下班高峰期产生的，所以该数据是个异常数据。不具有代表性。
#
# 风速对租赁情况的影响
windspeed_rentals = n_Bike_data.groupby(['windspeed'], as_index=True).agg({'casual':'max',
                                                                         'registered':'max',
                                                                         'count':'max'})
windspeed_rentals.plot(title = '不同风速下每小时最大使用量')
plt.show()

# 可以看到租赁数量随风速越大租赁数量越少，在风速超过30的时候明显减少，
# 但风速在风速40左右却有一次反弹，按常理来说风速大的话骑单车比较费劲，所以单车租赁数量也会减少，可以看下反弹原因

df2=n_Bike_data[Bike_data['windspeed']>40]
df2=df2[df2['count']>400]
# 可以发现这个时候时间在下班高峰期，是个异常值，不具有代表性。
#
# 日期对租赁数量的影响
# 考虑到相同日期是否工作日，星期几，以及所属年份等信息是一样的，把租赁数据按天求和，其它日期类数据取平均值

day_df = n_Bike_data.groupby(['date'], as_index=False).agg({'casual':'sum','registered':'sum',
                                                          'count':'sum', 'workingday':'mean',
                                                          'weekday':'mean','holiday':'mean',
                                                          'year':'mean'})

# 工作日 由于工作日和休息日的天数差别，对工作日和非工作日租赁数量取了平均值，对一周中每天的租赁数量求和
workingday_df=day_df.groupby(['workingday'], as_index=True).agg({'casual':'mean',
                                                                 'registered':'mean'})
workingday_df_0 = workingday_df.loc[0]
workingday_df_1 = workingday_df.loc[1]

# plt.axes(aspect='equal')
fig = plt.figure(figsize=(8,6))
plt.subplots_adjust(hspace=0.5, wspace=0.2)     #设置子图表间隔
grid = plt.GridSpec(2, 2, wspace=0.5, hspace=0.5)   #设置子图表坐标轴 对齐

plt.subplot2grid((2,2),(1,0), rowspan=2)
width = 0.3       # 设置条宽

p1 = plt.bar(workingday_df.index,workingday_df['casual'], width)
p2 = plt.bar(workingday_df.index,workingday_df['registered'],
             width,bottom=workingday_df['casual'])
plt.title('每天的平均租车数量')
plt.xticks([0,1], ('工作日', '非工作日'),rotation=20)
plt.legend((p1[0], p2[0]), ('casual', 'registered'))

plt.subplot2grid((2,2),(0,0))
plt.pie(workingday_df_0, labels=['casual','registered'], autopct='%1.1f%%',
        pctdistance=0.6 , labeldistance=1.35 , radius=1.3)
plt.axis('equal')
plt.title('工作日')

plt.subplot2grid((2,2),(0,1))
plt.pie(workingday_df_1, labels=['casual','registered'], autopct='%1.1f%%',
        pctdistance=0.6 , labeldistance=1.35 , radius=1.3)
plt.title('非工作日')
plt.axis('equal')

plt.show()

weekday_df= day_df.groupby(['weekday'], as_index=True).agg({'casual':'mean', 'registered':'mean'})
weekday_df.plot.bar(stacked=True , title = '一周之内每天租车数量')
plt.show()
# 对比图可发现：

# 工作日会员用户出行数量较多，临时用户出行数量较少；
# 周末会员用户租赁数量降低，临时用户租赁数量增加。
# 节假日 由于节假日在一年中数量占比非常少，先来看一每年的节假日下有几天：

holiday_coun=day_df.groupby('year', as_index=True).agg({'holiday':'sum'})
# 假期的天数占一年天数的份额十分少，所以对假期和非假期取日平均值

holiday_df = day_df.groupby('holiday',as_index=True).agg({'casual':'mean', 'registered':'mean'})
holiday_df.plot.bar(stacked=True , title = '节假日与非节假日每天租车量的平均值')

# 节假日会员或非会员使用量都比非节假日多，符合规律。

#                                                                  预测性分析
# 选择特征值
# 根据前面的观察，决定将时段（hour）、温度（temp）、湿度（humidity）、年份（year）、月份（month）、季节（season）、天气等级（weather）、
# 风速（windspeed_rfr）、星期几（weekday）、是否工作日（workingday）、是否假日（holiday），11项作为特征值。

Bike=n_Bike_data[['year','month','hour','weekday','workingday','holiday','temp','humidity',
                 'season','weather','windspeed_rfr','count']]

y = Bike['count']
X = Bike.drop(['count'],axis=1).select_dtypes(exclude=['object'])

# 训练集验证集分离

from sklearn.model_selection import train_test_split
train_X, test_X, train_y, test_y = train_test_split(X.values, y.values, test_size=0.3)

lgb_train = lgb.Dataset(train_X, train_y)
lgb_eval = lgb.Dataset(test_X, test_y, reference=lgb_train)

# 模型参数选择
#找到最优迭代次数
params = {
    'boosting_type': 'gbdt',
    'objective': 'regression',

    'learning_rate': 0.3,
    'num_leaves': 50,
    'max_depth': 17,

    'subsample': 0.8,
    'colsample_bytree': 0.8,
    }
cv_results = lgb.cv(
    params, lgb_train, num_boost_round=1000, nfold=5, stratified=False, metrics='rmse',
    early_stopping_rounds=50, verbose_eval=50,  seed=0)

print('best n_estimators:', len(cv_results['rmse-mean']))
print('best cv score:', cv_results['rmse-mean'][-1])

# 找到使模型效果达到最优的学习率、最大深度、最大叶子数、建树的特征选择比例
model_lgb = lgb.LGBMRegressor(objective='regression',
                              boosting_type='gbdt',
                              n_estimators=81,
                              metric='rmse')

params_test1 = {
    'max_depth': range(10, 30, 5),
    'num_leaves': range(50, 170, 30),
    'learning_rate': [0.3, 0.25, 0.2, 0.15, 0.1, 0.05, 0.01],
    'feature_fraction': [0.5, 0.6, 0.7, 0.8, 0.9],
    'bagging_fraction': [0.6, 0.7, 0.8, 0.9, 1.0]

}
gsearch1 = GridSearchCV(estimator=model_lgb, param_grid=params_test1, scoring='neg_mean_squared_error', cv=5, verbose=1,
                        n_jobs=4)
gsearch1.fit(train_X, train_y)

#最优参数
gsearch1.best_params_

gsearch1.best_estimator_

# 参数
params = {
    'task': 'train',
    'boosting_type': 'gbdt',  # 设置提升类型
    'objective': 'regression',  # 目标函数
    'metric': {'mae'},  # 评估函数
    'num_leaves':110
    ,  # 叶子节点数
    'max_depth':10,
    'learning_rate': 0.1,  # 学习速率
    'feature_fraction': 0.8,  # 建树的特征选择比例
    'bagging_fraction': 0.6,  # 建树的样本采样比例
    'bagging_freq': 10,  # k 意味着每 k 次迭代执行bagging
}
# 调用LightGBM模型，使用训练集数据进行训练（拟合）
my_model = lgb.train(params, lgb_train, num_boost_round=531, valid_sets=lgb_eval, early_stopping_rounds=200)
# 使用模型对验证集数据进行验证
predictions = my_model.predict(test_X, num_iteration=my_model.best_iteration)
# 对模型的预测结果进行评判
print("预测的均平均绝对误差是(mae): " + str(mean_absolute_error(predictions, test_y)))
print('预测的均方误差是(rmse):', mean_squared_error(predictions, test_y)**0.5)

test=Bike_data[(Bike_data['casual']==-1)&(Bike_data['registered']==-1)&(Bike_data['count']==-1)][['year','month','hour','weekday','workingday','holiday','temp','humidity',
                 'season','weather','windspeed_rfr']]

lgb_eval = lgb.Dataset(test,reference=lgb_train)
pre = my_model.predict(test, num_iteration=my_model.best_iteration)
test['pre']=pre
print(test)
