import os
import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.metrics import r2_score
import matplotlib.pylab as plt
from sklearn.metrics import mean_squared_error #均方误差
from sklearn.metrics import mean_absolute_error #平方绝对误差
from sklearn.metrics import r2_score#R square
import json
import lightgbm as lgbm
import pandas as pd
from sklearn import model_selection
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.metrics import mean_squared_error
import pickle

from compnents.material_plant_pearson import oneMaterialOnAllPlantPearson


def fit_lgbm(x_train, y_train, x_valid, y_valid,num, params: dict=None, verbose=100):
	#判断是否有训练好的模型，如果有的话直接加载，否则重新训练
    if os.path.isfile(f'../input/mlb-models/model_{num}.pkl'):
        with open(f'../input/mlb-models/model_{num}.pkl', 'rb') as fin:
            model = pickle.load(fin)
    else:
        oof_pred = np.zeros(len(y_valid), dtype=np.float32)
        model = lgbm.LGBMRegressor(**params)
        model.fit(x_train, y_train,
            eval_set=[(x_valid, y_valid)],
            early_stopping_rounds=verbose,
            verbose=verbose)
        #保存训练好的模型
        with open(f'model_{num}.pkl', 'wb') as handle:
            pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
    oof_pred = model.predict(x_valid) #对验证集进行预测
    score = mean_absolute_error(oof_pred, y_valid)#将预测结果与真是结果进行比较
    rmse = mean_squared_error(oof_pred, y_valid) ** 0.5
    r2 = r2_score(oof_pred, y_valid)
    print('mae:', score)
    print('rmse:', rmse)
    print('r2:', r2)

    return oof_pred, model, score

#lightgbm训练的参数：注意，上面的（**params）中的**必须写
params = {
'boosting_type': 'gbdt',
'objective':'mae',
'n_jobs':8,
'subsample': 0.5,
'subsample_freq': 1,
'learning_rate': 0.01,
'num_leaves': 2**11-1,
'min_data_in_leaf': 2**12-1,
'feature_fraction': 0.5,
'max_bin': 100,
'n_estimators': 2500,
'boost_from_average': False,
"random_seed":1,
}

material = '000000000070251989'
#data = data_extraction.get_df_from_db(material)
pearson, data = oneMaterialOnAllPlantPearson(material)
print(data)
#data=pd.read_excel(r'C:\Users\Administrator\Desktop\diabetes.xlsx')
#predictors=data.columns[:-1]
predictors = data['quantity']
data = data.iloc[:, 0:]
target = data.quantity
#data=data.drop(['AGE','SEX'],axis=1)
#拆分为训练集和测试集

features = data.iloc[:, 1:34]

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2)
# x_train,x_test,y_train,y_test=model_selection.train_test_split(data.iloc[:, 1:34], data.quantity,
#                                                         test_size=0.25,train_size=0.75)

# 加载你的数据
# print('Load data...')
# df_train = pd.read_csv('../regression/regression.train', header=None, sep='\t')
# df_test = pd.read_csv('../regression/regression.test', header=None, sep='\t')
#
# y_train = df_train[0].values
# y_test = df_test[0].values
# X_train = df_train.drop(0, axis=1).values
# X_test = df_test.drop(0, axis=1).values

# 创建成lgb特征的数据集格式
lgb_train = lgbm.Dataset(X_train, y_train)  # 将数据保存到LightGBM二进制文件将使加载更快
lgb_eval = lgbm.Dataset(X_test, y_test, reference=lgb_train)  # 创建验证数据

#将第一步整理的输入数据塞入模型即可
oof1, model1, score1 = fit_lgbm(
    X_train, y_train,
    X_test, y_test,1,
    params
 )
