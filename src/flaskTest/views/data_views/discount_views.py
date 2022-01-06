from flaskTest.db.discount import *
from flaskTest.preprocess.discount.discount_preprocess import *


def get_discount_by_material(material, start_time, end_time):

    # 数据库抽取数据
    discount = get_discount_from_db_by_material(material, start_time, end_time)

    discount = preprocess_discount(discount)

    # print(discount.info)
    # print(discount.describe().T)
    # print(discount.dtypes)
    # print(discount['discount_rate'])
    return discount

def get_discount_by_category(category, start_time, end_time):
    # 数据库抽取数据
    discount = ''



    # print(discount.info)
    # print(discount.describe().T)
    # print(discount.dtypes)
    # print(discount['discount_rate'])
    return discount


# get_discount_by_material('000000000070251989')