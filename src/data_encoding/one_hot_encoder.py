from sklearn.preprocessing import OneHotEncoder
import pandas as pd


def one_hot_encoding(df, column):
    df_new = df.copy()

    value_sets = df_new[column].unique()
    for value_unique in value_sets:
        col_name_new = column + '_' + value_unique
        df_new[col_name_new] = (df_new[column] == value_unique)
        df_new[col_name_new] = df_new[col_name_new].astype('int32')
    return df_new


    # one_hot = OneHotEncoder()
    #
    # data_temp = pd.DataFrame(one_hot.fit_transform(df[col_set]).toarray(),
    #
    #     columns = one_hot.get_feature_names(col_set), dtype = 'int')
    #
    # df_onehot = pd.concat((df, data_temp), axis=1)  # 也可以用merge,join
    # return df_onehot



