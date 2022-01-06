from flaskTest.preprocess.common_ouliers_preprocess import *
#from data_preprocess.missing_value_processing import *
import numpy as np

def total_missing_value_process(total):

    ##删除number_station、number_store为空的行
    total.replace('None', np.nan, inplace=True)
    col_set = ['number_station', 'number_store','type07']
    total.dropna(axis=0, subset=col_set, inplace=True)
    # 缺失值处理
    missing_value_fill_str_NP(total, 'promotion_type')
    missing_value_fill_num_1(total, 'discount_rate2')
    return total

def plant_and_poi_missing_value_process(plant_and_poi):

    plant_and_poi = plant_and_poi.fillna(0)

    return plant_and_poi


