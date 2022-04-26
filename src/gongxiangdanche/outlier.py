import numpy as np

# 删除异常值
def delete_outlier(df, col):

    # 判断count是否为离群值（异常值），如果不是则不选取（删除）
    df_WithoutOutliers = df[np.abs(df[col]) >= df[col].mean() + (3 * df[col].std())]

    return df_WithoutOutliers

# 平均值减去三倍标准差
def replace_val(val, mean, std):
    if val > mean + 3*std:
        return abs(mean + 3*std)
    elif val < mean - 3*std:
        return abs(mean - 3*std)
    return val

# 替换离群值
def replace_outlier(df, cols):
    for col in cols:
        mean = df[col].mean()
        std_dev = df[col].std(axis=0)
        df[col] = df[col].map(lambda x: replace_val(x, mean, abs(std_dev)))

    return df



