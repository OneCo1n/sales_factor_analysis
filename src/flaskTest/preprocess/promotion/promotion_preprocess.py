
from data_preprocess.missing_value_processing import missing_value_check, nan_value_check
from flaskTest.preprocess.promotion.promotion_col_name import promotion_col_name_exg
from flaskTest.preprocess.promotion.promotion_dtypes import *
from flaskTest.preprocess.promotion.promotion_feature_select import promotion_feature_select
from flaskTest.preprocess.promotion.promotion_outliers import promotion_discount_rate_outliers_replace
from gongxiangdanche.density import show_density, show_discount_density


def preprocess_promotion(promotion):


    # 缺失值检测
    missing_value_check(promotion)
    # nan_value_check(promotion)
    # 更改列名
    promotion = promotion_col_name_exg(promotion)
    # 类型转换
    promotion = promotion_dtypes_exg(promotion)

    promotion = promotion_discount_rate_outliers_replace(promotion)
    promotion = promotion_feature_select(promotion)
    show_discount_density(promotion, 'discount_rate2')


    return promotion


def preprocess_promotion_desc(promotion):
    promotion['plant'] = promotion['plant'].astype('str')
    promotion['date'] = promotion['date'].astype('datetime64')
    promotion = promotion_discount_rate_outliers_replace(promotion)



    return promotion


