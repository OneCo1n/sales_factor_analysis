from data_preprocess.missing_value_processing import missing_value_check
from flaskTest.preprocess.discount.discount_col_name import discount_col_name_exg
from flaskTest.preprocess.discount.discount_dtypes import *
from gongxiangdanche.outlier import *


def preprocess_discount(discount):
    # 缺失值检测
    # missing_value_check(discount)
    # 更改列名
    discount = discount_col_name_exg(discount)
    # 类型转换
    discount = discount_dtypes_exg(discount)
    # 异常值处理

    # 缺失值

    # 异常值


    return discount