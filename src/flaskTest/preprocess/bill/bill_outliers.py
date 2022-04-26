from gongxiangdanche.outlier import *


def bill_outliers_replace(bill, outlier_col):
    bill = replace_outlier(bill, outlier_col)

    return bill

def bill_outliers_delete(bill, outlier_col):
    bill = delete_outlier(bill, outlier_col)

    return bill