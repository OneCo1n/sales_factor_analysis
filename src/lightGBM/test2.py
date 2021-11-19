# -*- coding: utf-8 -*-
# author: Yu Sun

import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import plot_importance
from compnents.material_plant_pearson import oneMaterialOnAllPlantPearson

params = {
    'task': 'train',
    'boosting_type': 'gbdt',  # GBDT算法为基础
    'objective': 'regression',  # 因为要完成预测用户是否买单行为，所以是binary，不买是0，购买是1
    'metric': 'auc',  # 评判指标
    'max_bin': 255,  # 大会有更准的效果,更慢的速度
    'learning_rate': 0.1,  # 学习率
    'num_leaves': 64,  # 大会更准,但可能过拟合
    'max_depth': -1,  # 小数据集下限制最大深度可防止过拟合,小于0表示无限制
    'feature_fraction': 0.8,  # 防止过拟合
    'bagging_freq': 5,  # 防止过拟合
    'bagging_fraction': 0.8,  # 防止过拟合
    'min_data_in_leaf': 21,  # 防止过拟合
    'min_sum_hessian_in_leaf': 3.0,  # 防止过拟合
    'header': True  # 数据集是否带表头
}


# 训练模型并预测
def train_predict_model(model_file='./model.txt'):
    #dataset = pd.read_csv("./data/train.csv")  # 训练集


    material = '000000000070251989'
    # data = data_extraction.get_df_from_db(material)
    pearson, dataset = oneMaterialOnAllPlantPearson(material)
    print(dataset)
    d_x = dataset.iloc[:, 2:].values
    d_y = dataset['quantity'].values

    predictors = dataset['quantity']

    data = dataset.iloc[:, 0:]
    target = data.quantity
    # data=data.drop(['AGE','SEX'],axis=1)
    # 拆分为训练集和测试集

    # features = data.iloc[:, 1:34]
    # dataset_future = pd.read_csv("./data/test.csv")  # 测试集（用于在线提交结果）
    # d_future_x = dataset_future.iloc[:, 2:].values

    train_X, valid_X, train_Y, valid_Y = train_test_split(
        d_x, d_y, test_size=0.2, random_state=2)  # 将训练集分为训练集+验证集

    lgb_train = lgb.Dataset(train_X, label=train_Y)
    lgb_eval = lgb.Dataset(valid_X, label=valid_Y, reference=lgb_train)
    print("Training...")
    bst = lgb.train(
        params,
        lgb_train,
        categorical_feature=list(range(1, 17)),  # 指明哪些特征的分类特征
        valid_sets=[lgb_eval],
        num_boost_round=500,
        early_stopping_rounds=200)

    oof_pred = bst.predict(valid_X)  # 对验证集进行预测
    score = mean_absolute_error(oof_pred, valid_Y)  # 将预测结果与真是结果进行比较
    rmse = mean_squared_error(oof_pred, valid_Y) ** 0.5
    r2 = r2_score(oof_pred, valid_Y)
    print('mae:', score)
    print('rmse:', rmse)
    print('r2:', r2)

    print("Saving Model...")
    bst.save_model(model_file)  # 保存模型
    # print("Predicting...")
    # predict_result = bst.predict(d_future_x)  # 预测的结果在0-1之间，值越大代表预测用户购买的可能性越大
    #
    # return predict_result
    plt.figure(figsize=(12, 6))
    plot_importance(bst)
    #lgb.plot_importance(bst)
    plt.title("Featurertances")
    plt.show()
    return dataset, bst

# 评估选取的各特征的重要度（画图显示）
def plot_feature_importance(dataset, model_bst):
    list_feature_name = list(dataset.columns[2:])
    list_feature_importance = list(model_bst.feature_importance(
        importance_type='split', iteration=-1))
    dataframe_feature_importance = pd.DataFrame(
        {'feature_name': list_feature_name, 'importance': list_feature_importance})
    print(dataframe_feature_importance)
    x = range(len(list_feature_name))
    plt.xticks(x, list_feature_name, rotation=90, fontsize=14)
    plt.plot(x, list_feature_importance)
    for i in x:
        plt.axvline(i)
    plt.show()



if __name__ == "__main__":
    dataset, model = train_predict_model()
    plot_feature_importance(dataset, model)