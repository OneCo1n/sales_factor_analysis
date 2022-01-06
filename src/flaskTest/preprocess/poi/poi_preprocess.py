from data_preprocess.missing_value_processing import missing_value_check, nan_value_check
from flaskTest.preprocess.plant.plant_col_name import *
from flaskTest.preprocess.plant.plant_dtypes import *
from flaskTest.preprocess.poi.poi_dtypes import poi_dtypes_exg
from gongxiangdanche.outlier import *


def preprocess_poi(poi):
    # 缺失值检测
    # missing_value_check(poi)
    # nan_value_check(poi)
    # 更改列名
    # plant_col_name_exg(plant)
    # 类型转换
    poi = poi_dtypes_exg(poi)




    return poi
