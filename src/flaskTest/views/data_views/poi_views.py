from flaskTest.db.poi import *
from flaskTest.preprocess.poi.poi_preprocess import *


def get_poi():

    # 数据库抽取数据
    poi = get_poi_from_db()
    # 数据预处理
    poi = preprocess_poi(poi)

    # print(poi.info)
    # print(poi.describe().T)
    # print(poi.dtypes)

    return poi


# get_poi()