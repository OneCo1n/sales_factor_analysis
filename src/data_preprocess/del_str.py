import warnings
warnings.filterwarnings('ignore')

#删除df中col字段中的 m2、M2、平米 字符串
def del_str_m2(df, col):
    df[col].replace('m2', '', regex=True, inplace=True)
    df[col].replace('M2', '', regex=True, inplace=True)
    df[col].replace('平米', '', regex=True, inplace=True)

#删除df中col字段中的 - 字符
def del_str_negative_sign(df, col):
    df[col].replace('-', '', regex=True, inplace=True)