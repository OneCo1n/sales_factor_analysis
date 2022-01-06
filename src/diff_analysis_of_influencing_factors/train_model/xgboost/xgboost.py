from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_error  # 均方误差
from sklearn.metrics import mean_absolute_error  # 平方绝对误差
from sklearn.metrics import r2_score  # R square
from sklearn.datasets import load_iris
import xgboost as xgb
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import explained_variance_score


def use_xgboost(x_train, x_test, y_train, y_test, target_name):
    # 训练模型
    model = xgb.XGBClassifier(max_depth=5, learning_rate=0.1, n_estimators=160, silent=True, objective='multi:softmax')
    model.fit(x_train, y_train)

    # 对测试集进行预测
    y_pred = model.predict(x_test)

    # 计算准确率
    accuracy = explained_variance_score(y_test, y_pred)
    print('accuracy:%2.f%%' % (accuracy * 100))

    # 评估模型
    print('[info : lightGBM] The mse of prediction is:', mean_squared_error(y_test, y_pred))
    print('[info : lightGBM] The rmse of prediction is:',
          mean_squared_error(y_test, y_pred) ** 0.5)  # 计算真实值和预测值之间的均方根误差
    print('[info : lightGBM] the r2 of prediction is:', r2_score(y_test, y_pred))

    importance = model.feature_importances_
    feature_name = model._Booster.feature_names
    importance_dict = dict()

    for i, label in enumerate(feature_name):
        importance_dict[label] = float(importance[i])

    importance_dict = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)

    importance_dict = dict(importance_dict)

    # estimator = lgb.LGBMRegressor(num_leaves=31)
    # param_grid = {
    #     'learning_rate': [0.01, 0.1, 1],
    #     'n_estimators': [20, 40]
    # }
    # gbm = GridSearchCV(estimator, param_grid)
    # gbm.fit(X_train, y_train)
    # print('[info : lightGBM] Best parameters found by grid search are:', gbm.best_params_)
    return importance_dict


