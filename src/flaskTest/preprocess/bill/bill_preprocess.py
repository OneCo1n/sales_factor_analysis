from data_preprocess.missing_value_processing import missing_value_check
from flaskTest.preprocess.bill.bill_dtypes import bill_dtypes_exg
from flaskTest.preprocess.bill.bill_outliers import bill_outliers_replace


def preprocess_bill(bill):
    # 缺失值检测
    # missing_value_check(bill)
    # 类型转换
    bill = bill_dtypes_exg(bill)
    # 异常值处理
    outlier_col = ["quantity"]
    bill = bill_outliers_replace(bill, outlier_col)


    return bill