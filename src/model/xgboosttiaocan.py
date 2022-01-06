import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import model_selection, metrics
from sklearn.model_selection import GridSearchCV

import matplotlib.pylab as plt

from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 4

train = pd.read_csv('train_modified.csv')
test = pd.read_csv('test_modified.csv')

print(train.shape, test.shape)

target='Disbursed'
IDcol = 'ID'

print(train['Disbursed'].value_counts())

# 建模与交叉验证
# 写一个大的函数完成以下的功能
#
# 1.数据建模
# 2.求训练准确率
# 3.求训练集AUC
# 4.根据xgboost交叉验证更新n_estimators
# 5.画出特征的重要度

# test_results = pd.read_csv('test_results.csv')
# def modelfit(alg, dtrain, dtest, predictors, useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
#     if useTrainCV:
#         xgb_param = alg.get_xgb_params()
#         xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain[target].values)
#         xgtest = xgb.DMatrix(dtest[predictors].values)
#         cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
#                           early_stopping_rounds=early_stopping_rounds, verbose_eval=False)
#         alg.set_params(n_estimators=cvresult.shape[0])
#
#     # 建模
#     alg.fit(dtrain[predictors], dtrain['Disbursed'], eval_metric='auc')
#
#     # 对训练集预测
#     dtrain_predictions = alg.predict(dtrain[predictors])
#     dtrain_predprob = alg.predict_proba(dtrain[predictors])[:, 1]
#
#     # 输出模型的一些结果
#     print("\n关于现在这个模型")
#
#     print("准确率 : %.4g" % metrics.accuracy_score(dtrain['Disbursed'].values, dtrain_predictions))
#
#     print("AUC 得分 (训练集): %f" % metrics.roc_auc_score(dtrain['Disbursed'], dtrain_predprob))
#
#     feat_imp = pd.Series(alg.get_booster().get_fscore()).sort_values(ascending=False)
#     feat_imp.plot(kind='bar', title='Feature Importances')
#     plt.ylabel('Feature Importance Score')
#     plt.show()
#
#
# predictors = [x for x in train.columns if x not in [target, IDcol]]
# xgb1 = XGBClassifier(
#         learning_rate =0.1,
#         n_estimators=1000,
#         max_depth=5,
#         min_child_weight=1,
#         gamma=0,
#         subsample=0.8,
#         colsample_bytree=0.8,
#         objective= 'binary:logistic',
#         nthread=4,
#         scale_pos_weight=1,
#         seed=27)
# modelfit(xgb1, train, test, predictors)

def modelfit(dtrain, dtest, predictors,  cv_folds=5, early_stopping_rounds=50):
    xgb1 = XGBClassifier(
        learning_rate=0.1,
        n_estimators=1000,
        max_depth=5,
        min_child_weight=1,
        gamma=0,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='binary:logistic',
        nthread=4,
        scale_pos_weight=1,
        seed=27)

    xgb_param = xgb1.get_xgb_params()

    xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain[target].values)
    xgtest = xgb.DMatrix(dtest[predictors].values)
    cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=xgb1.get_params()['n_estimators'], nfold=cv_folds,
                          early_stopping_rounds=early_stopping_rounds, verbose_eval=False)
    xgb1.set_params(n_estimators=cvresult.shape[0])

    # 建模
    xgb1.fit(dtrain[predictors], dtrain['Disbursed'], eval_metric='auc')

    # 对训练集预测
    dtrain_predictions = xgb1.predict(dtrain[predictors])
    dtrain_predprob = xgb1.predict_proba(dtrain[predictors])[:, 1]

    # 输出模型的一些结果
    print("\n关于现在这个模型")

    print("准确率 : %.4g" % metrics.accuracy_score(dtrain['Disbursed'].values, dtrain_predictions))

    print("AUC 得分 (训练集): %f" % metrics.roc_auc_score(dtrain['Disbursed'], dtrain_predprob))

    feat_imp = pd.Series(xgb1.get_booster().get_fscore()).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')
    plt.show()


predictors = [x for x in train.columns if x not in [target, IDcol]]

modelfit(train, test, predictors)