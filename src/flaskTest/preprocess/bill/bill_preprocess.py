from data_preprocess.missing_value_processing import missing_value_check
from flaskTest.preprocess.bill.bill_dtypes import bill_dtypes_exg
from flaskTest.preprocess.bill.bill_outliers import bill_outliers_replace, bill_outliers_delete
import numpy as np

from gongxiangdanche.density import show_density


def preprocess_bill(bill):
    # 缺失值检测
    missing_value_check(bill)
    # 类型转换
    bill = bill_dtypes_exg(bill)
    show_density(bill, 'quantity')
    # 对数转换
    log_col = ["quantity"]
    bill = bill_log(bill, log_col)

    # 异常值处理
    outlier_col = ["quantity"]
    bill = bill_outliers_replace(bill, outlier_col)

    # show_density(bill, 'quantity')
    # bill = bill_outliers_replace(bill, outlier_col)
    # bill = bill_outliers_replace(bill, outlier_col)
    # bill = bill_outliers_replace(bill, outlier_col)
    # bill = bill_outliers_delete(bill, outlier_col)


    return bill

def bill_log(df, log_col):
    for i in log_col:
        df_x = df.copy()
        df_x.loc[:, i] = np.where((df_x.loc[:, i] == 0), df_x.loc[:, i], np.log2(df_x.loc[:, i]))
        #df[i] = df[i].map(lambda x: np.log2(x))  # 进行对数转换 以 e 为底 想以其他为底可以变换如 np.log10(x)
        # df[(df[i] > 0)][i] = df[(df[i] > 0)][i].map(lambda x: np.log2(x))

    return df_x