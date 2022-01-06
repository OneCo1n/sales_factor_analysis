from data_preprocess.missing_value_processing import missing_value_check, nan_value_check
from flaskTest.preprocess.plant.plant_area_str import plant_area_m2_delete
from flaskTest.preprocess.plant.plant_col_name import *
from flaskTest.preprocess.plant.plant_dtypes import *
from flaskTest.preprocess.plant.plant_feature_select import plant_feature_select
from gongxiangdanche.outlier import *


def preprocess_plant(plant):
    # 缺失值检测
    #missing_value_check(plant)
    nan_value_check(plant)
    # 更改列名
    # plant_col_name_exg(plant)
    # 删除面积中的多余字符串
    plant = plant_area_m2_delete(plant)
    # 类型转换
    plant = plant_dtypes_exg(plant)
    plant = plant_feature_select(plant)
    # 异常值处理

    # 缺失值

    # 异常值
    return plant
