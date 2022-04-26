
from flaskTest.db.promotion import *

from flaskTest.preprocess.promotion.promotion_preprocess import preprocess_promotion, preprocess_promotion_desc


def get_promotion_by_material(material, start_time, end_time):

    # 数据库抽取数据
    promotion = get_promotion_from_db_by_material(material, start_time, end_time)

    promotion = preprocess_promotion(promotion)


    # print(promotion.info)
    # print(promotion.describe().T)
    # print(promotion.dtypes)
    # print(promotion['discount_rate2'].min())

    return promotion

def get_promotion_by_all_material_and_plant_class(start_time, end_time, plant_class):
    # 数据库抽取数据
    promotion = get_promotion_from_db_by_all_material_and_plant_class(start_time, end_time, plant_class)

    promotion = preprocess_promotion(promotion)


    # print(promotion.info)
    # print(promotion.describe().T)
    # print(promotion.dtypes)
    # print(promotion['discount_rate2'].min())

    return promotion

def get_promotion_by_category(category, start_time, end_time):
    # 数据库抽取数据
    promotion = get_promotion_from_db_by_category(category, start_time, end_time)

    promotion = preprocess_promotion(promotion)

    # print(promotion.info)
    # print(promotion.describe().T)
    # print(promotion.dtypes)
    # print(promotion['discount_rate2'].min())

    return promotion

def get_promotion_desc_by_material(material, start_time, end_time):
    # 数据库抽取数据
    promotion = get_promotion_desc_from_db_by_material(material, start_time, end_time)

    promotion = preprocess_promotion_desc(promotion)

    # print(promotion.info)
    # print(promotion.describe().T)
    # print(promotion.dtypes)
    #print(promotion['discount_rate2'].min())

    return promotion


def get_promotion_desc_by_category(category, start_time, end_time):
    # 数据库抽取数据
    promotion = get_promotion_desc_from_db_by_category(category, start_time, end_time)

    promotion = preprocess_promotion_desc(promotion)

    # print(promotion.info)
    # print(promotion.describe().T)
    # print(promotion.dtypes)
    # print(promotion['discount_rate2'].min())

    return promotion

# get_promotion_by_material('000000000070251989')

