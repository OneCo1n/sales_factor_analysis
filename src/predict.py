# coding=utf-8 ##以utf-8编码储存中文字符
from sklearn.model_selection import GridSearchCV   #Perforing grid search
import pandas as pd
from sklearn.metrics import mean_squared_error
import numpy as np
from scipy.stats import mode
import warnings
from sklearn.model_selection import train_test_split
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
import operator
import matplotlib.pyplot as plt
import matplotlib
# 根据历史的每天销量预测未来的销量，之前稍微统计，每个月的总量基本是不变的。
#
# 效果其实和求均值差不多
warnings.filterwarnings('ignore')

def mode_function(df):
    df = df.astype(int)
    counts = mode(df)
    return counts[0][0]

def score(y_test,y_pred):
    print(y_test)
    print(y_pred)
    return 1.0 / (1.0 + np.sqrt(mean_squared_error(y_test, y_pred)))

train = pd.read_csv('input/train.csv')
print('训练集总数',train.shape)
print('字段',train.columns)
sub = pd.read_csv('input/result.csv')
sub['SaleM'] = sub['日期'].map(lambda x:int(str(x)[5:6]))
sub['SaleD'] = sub['日期'].map(lambda x:int(str(x)[6:]))
#sub['销售日期_D'] = sub['日期'].map(lambda x:str(x)[-2:])
sub['SaleD'] = sub['SaleD'] .astype(int)
sub['SaleM'] = sub['SaleM'] .astype(int)
result = sub.copy()
#print(sub)
sub_train = sub[['编码', 'SaleM', 'SaleD', '销量', '日期']]
#print("sub_train:")
#print(sub_train)

train['SaleM'] = train['销售日期'].map(lambda x:int(str(x)[5:6]))
train['SaleD'] = train['销售日期'].map(lambda x:int(str(x)[6:]))
train['SaleD'] = train['SaleD'] .astype(int)
train['SaleM'] = train['SaleM'] .astype(int)
train.drop(['大类名称', '中类名称', '小类名称', '销售月份', '商品编码', '小类编码', '单位', '销售日期', '规格型号', 'custid', '商品类型', '商品单价', '销售金额', '销售数量', '是否促销'],axis=1,inplace=True)



def cal_w(row):
    if row.SaleM == 1:
        return (row.SaleD + 3) % 7
    elif row.SaleM == 2:
        return (row.SaleD + 6) % 7
    elif row.SaleM == 3:
        return (row.SaleD + 6) % 7
    elif row.SaleM == 4:
        return (row.SaleD + 2) % 7
    elif row.SaleM == 5:
        return (row.SaleD + 4) % 7


train['SaleW'] = train.apply(cal_w, axis=1)
train['SaleW'] = train['SaleW'].astype(int)

def cal_wn(row):
    if row.SaleM == 1:
        return (row.SaleD + 2) / 7 + 1
    elif row.SaleM == 2:
        return (row.SaleD + 5) / 7 + 5
    elif row.SaleM == 3:
        return (row.SaleD + 5) / 7 + 9
    elif row.SaleM == 4:
        return (row.SaleD + 1) / 7 + 14
    elif row.SaleM == 5:
        return (row.SaleD + 3) / 7 + 18


train['SaleWn'] = train.apply(cal_wn, axis=1)
train['SaleWn'] = train['SaleWn'].astype(int)

sub_train['SaleW'] = sub_train.apply(cal_w, axis=1)
sub_train['SaleW'] = sub_train['SaleW'].astype(int)

sub_train['SaleWn'] = sub_train.apply(cal_wn, axis=1)
sub_train['SaleWn'] = sub_train['SaleWn'].astype(int)
#sub_train['SaleWn'] = sub_train.apply(cal_wn, axis=1)
#sub_train['SaleWn'] = sub_train['SaleWn'].astype(int)
#sub_train.drop(['SaleM', 'SaleD'], axis=1, inplace=True)
sub_train.drop(['日期'],axis=1,inplace=True)
train['is_buy'] = 1
train['is_buy'] = train['is_buy'].astype(int)
train_count = train.groupby(['中类编码', 'SaleW', 'SaleWn', 'SaleD', 'SaleM'],as_index=False)['is_buy'].sum()
train_count2 = train.groupby(['大类编码', 'SaleW', 'SaleWn', 'SaleD', 'SaleM'],as_index=False)['is_buy'].sum()


train_count.rename(columns = {'is_buy':'销量'},inplace=True)
train_count.rename(columns = {'中类编码':'编码'},inplace=True)
train_count['label'] = 1
train_count2.rename(columns = {'is_buy':'销量'},inplace=True)
train_count2.rename(columns = {'大类编码':'编码'},inplace=True)
train_count2['label'] = 0
train_count= pd.concat([train_count,train_count2])
train_count=train_count[train_count['SaleWn'] != 8]
indexs = train_count[(train_count.SaleM==2)&(train_count.SaleD==4)].index.tolist()
train_count = train_count.drop(indexs)


def cal_label(row):
    if row.编码 < 1000:
        return 0
    else:
        return 1


sub_train['label'] = sub_train.apply(cal_label, axis=1)

#print(train_count)
#print(sub_train)
#print(sub_train)
#print(train_count)

day_dummies_train  = pd.get_dummies(train_count['SaleW'], prefix='Day')
day_dummies_test  = pd.get_dummies(sub_train['SaleW'], prefix='Day')
train_count = train_count.join(day_dummies_train)
sub_train = sub_train.join(day_dummies_test)

#train_count.drop(['SaleW', 'SaleD', 'SaleM'], axis=1,inplace=True)
#sub_train.drop(['SaleW', 'SaleD', 'SaleM'], axis=1,inplace=True)
#print(train_count)
print(sub_train)

def rmspe(y, yhat):
    return np.sqrt(np.mean((yhat/y-1) ** 2))

def rmspe_xg(yhat, y):
    y = np.expm1(y.get_label())
    yhat = np.expm1(yhat)
    return "rmspe", rmspe(y,yhat)

def build_features(features):
#     features.extend(['编码', 'SaleWn','Day_0','Day_1','Day_2','Day_3','Day_4','Day_5','Day_6']) #0.43926 0.42849
#     features.extend(['编码', 'SaleWn', 'SaleW']) #0.336992 0.361086 0.351852
#     features.extend(['编码', 'SaleW','SaleD','SaleWn'])
     features.extend(['编码', 'SaleWn','Day_0','Day_1','Day_2','Day_3','Day_4','Day_5','Day_6','SaleW','SaleD','label'])


params = {"objective": "reg:linear",
          "booster" : "gbtree",
          "max_depth": 9,
          "subsample": 1.0,
          "eta": 0.3,
          "eval_metric": "rmse",
          "silent": 0,

          }
features = []
build_features(features)
num_boost_round = 100000
print("Train a XGBoost model")

#X_train = train_count[train_count['SaleWn'] < 14]      0.1523 并不如随机划分
#X_valid = train_count[train_count['SaleWn'] > 14]
X_train, X_valid = train_test_split(train_count, test_size=0.2)
y_train = X_train.销量
y_valid = X_valid.销量
dtrain = xgb.DMatrix(X_train[features], y_train)
dvalid = xgb.DMatrix(X_valid[features], y_valid)
watchlist = [(dtrain, 'train'), (dvalid, 'eval')]

#gsearch1 = GridSearchCV(estimator = XGBClassifier(         learning_rate =0.1,
#min_child_weight=1,  subsample=1,
# objective= 'reg:linear'),
# param_grid = param_test1,     scoring='roc_auc',iid=False)
#gsearch1.fit(dtrain,dvalid)
#print(gsearch1)
gbm = xgb.train(params, dtrain, num_boost_round, evals=watchlist, \
  early_stopping_rounds=100, verbose_eval=True)

print("Make predictions on the test set")
dtest = xgb.DMatrix(sub_train[features])
test_probs = gbm.predict(dtest)
# Make Submission
sub_ = pd.DataFrame({u'销量':list(test_probs)})
sub_['销量'] = sub_['销量']
sub_['销量'] = sub_['销量'].astype(int)

#print(sub_)
result = result[['编码','日期']]
# 单独看了看中类 去掉大于10000 就是全部需要预测的
result_sub_ = pd.concat([result[['编码','日期']],sub_['销量']],axis=1)

def cal_r(row):
    if row.销量 < 0:
        return 0
    else:
        return row.销量

result_sub_['销量'] = result_sub_.apply(cal_r, axis=1)
result_sub_.to_csv('./sample_1.csv', index=False)


#这里要先写xgb.map然后把编码改一下再读画图
#def create_feature_map(features):
#    outfile = open('xgb.fmap', 'w')
#    for i, feat in enumerate(features):
#        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
#    outfile.close()
#create_feature_map(features)

'''
importance = gbm.get_fscore(fmap='xgb.fmap')
importance = sorted(importance.items(), key=operator.itemgetter(1))

df = pd.DataFrame(importance, columns=['feature', 'fscore'])
df['fscore'] = df['fscore'] / df['fscore'].sum()

featp = df.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(6, 10))
plt.title('XGBoost Feature Importance')
plt.xlabel('relative importance')
fig_featp = featp.get_figure()
fig_featp.savefig('feature_importance_xgb.png', bbox_inches='tight', pad_inches=1)



'''