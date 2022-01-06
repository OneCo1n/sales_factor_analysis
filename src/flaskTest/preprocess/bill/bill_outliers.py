from gongxiangdanche.outlier import *


def bill_outliers_replace(bill, outlier_col):
    bill = replace_outlier(bill, outlier_col)

    return bill