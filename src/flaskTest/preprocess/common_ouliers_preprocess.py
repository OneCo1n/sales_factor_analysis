
import pandas as pd


def missing_value_check(df):
    #
    isnull_col = df.isnull().sum().sort_values(ascending=False) / len(df)

    #isnull_col[isnull_col >= 0.5]
    print("各特征缺失情况：")
    print(df.notnull())
    print("各特征缺失率：")
    print(isnull_col)
    print("---------------------------------\n显示每一列中有多少个缺失值：\n", df.isnull().sum())
    print("---------------------------------\n含有缺失值的记录：\n")
    print(df[df.isnull().values==True])

def nan_value_check(df):
    isna_col = df.isna().sum().sort_values(ascending=False) / len(df)

    # isnull_col[isnull_col >= 0.5]
    print("各特征nan情况：")
    print(df.notnull())
    print("各特征nan率：")
    print(isna_col)
    print("---------------------------------\n显示每一列中有多少个nan值：\n", df.isna().sum())
    print("---------------------------------\n含有nan的记录：\n")
    print(df[df.isna().values == True])


# 对df中col列的缺失值填充 0
def missing_value_fill_num_0(df, col):

    df[col].fillna(value=0, inplace=True)

#对df中col列的缺失值填充 1
def missing_value_fill_num_1(df, col):
    df[col].fillna(value=1, inplace=True)

#对df中col列的缺失值填充 众数
def missing_value_fill_mode(df, col):
    df[col].fillna(df[col].mode(), inplace=True)

def missing_value_fill_str_O(df, col):
    df[col].fillna('O', inplace=True)

def missing_value_fill_str_NP(df, col):
    df[col].fillna('NP', inplace=True)


def del_missing_value_row(df, col_set):
    df.dropna(axis=0, subset=col_set, inplace = True)

