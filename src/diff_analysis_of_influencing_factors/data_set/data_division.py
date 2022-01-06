from sklearn.model_selection import train_test_split


def model_data_division(data, target_name, test_size):

    data = data.iloc[:, 0:]
    target = data[target_name]

    # 拆分为训练集和测试集
    features = data.drop(labels=None, axis=1, index=None, columns=['plant', 'date', target_name], inplace=False)
    x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=test_size)

    return x_train, x_test, y_train, y_test