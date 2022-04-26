from diff_analysis_of_influencing_factors.data_set.data_division import model_data_division
from diff_analysis_of_influencing_factors.train_model.lasso.lasso import use_lasso
from diff_analysis_of_influencing_factors.train_model.xgboost.xgboost import use_xgboost


def do_analysis_use_lasso_xgboost(data, target_name):
    # 特征选择 数据集划分

    x_train, x_test, y_train, y_test = model_data_division(data, target_name, 0.2)
    # lasso模型进行预训练 返回系数为0 的特征
    rejected_features = use_lasso(x_train, x_test, y_train, y_test, target_name)
    # 完成特征筛选
    selected_data = features_select_from_rejected_list(data, rejected_features)
    # 对特征筛选后的数据集进行划分
    X_train, X_test, Y_train, Y_test = model_data_division(selected_data, target_name, 0.2)

    # 数据传入模型开始训练  获取分析结果
    mse, rmse, r2, mae, mape, importance_dict = use_xgboost(X_train, X_test, Y_train, Y_test, target_name)
    return mse, rmse, r2, mae, mape, importance_dict



def features_select_from_rejected_list(data, rejected_features):

    select_features = data.drop(labels=None, axis=1, index=None, columns=rejected_features, inplace=False)

    return select_features