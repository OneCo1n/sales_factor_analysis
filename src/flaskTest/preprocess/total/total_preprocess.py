from data_preprocess.missing_value_processing import missing_value_check, nan_value_check
from flaskTest.preprocess.total.merge import total_data_merge, plant_and_poi_merge, bill_and_plant_desc_merge, \
    bill_and_promotion_merge, total_category_data_merge
from flaskTest.preprocess.total.total_encoder import total_encoding, plant_and_poi_encoding
from flaskTest.preprocess.total.total_outliers import *



def preprocess_total(bill, discount, plant, poi, promotion):

    # 数据合并
    total = total_data_merge(bill, discount, plant, poi, promotion)
    # 删除某些缺失值
    total = total_missing_value_process(total)
    # 缺失值检测
    print("-------------------total缺失值检测----------------")
    # missing_value_check(total)
    # nan_value_check(total)
    # 删除无用特征

    # 特征编码
    total = total_encoding(total)

    return total

def preprocess_total_category(bill, discount, plant, poi, promotion):
    # 数据合并
    total = total_category_data_merge(bill, discount, plant, poi, promotion)
    # 删除某些缺失值
    total = total_missing_value_process(total)
    # 缺失值检测
    print("-------------------total缺失值检测----------------")
    # missing_value_check(total)
    # nan_value_check(total)
    # 删除无用特征
    total = total.drop(labels=None, axis=1, index=None,
                               columns=['material'],
                               inplace=False)


    # 特征编码
    total = total_encoding(total)

    return total


def preprocess_total_desc(bill, discount, plant, poi, promotion):

    # 数据合并
    total = total_data_merge(bill, discount, plant, poi, promotion)
    # 删除某些缺失值
    total = total_missing_value_process(total)
    # 缺失值检测
    print("-------------------total缺失值检测----------------")
    # missing_value_check(total)
    # nan_value_check(total)
    # 删除无用特征

    # 特征编码
    #total = total_encoding(total)

    return total

def preprocess_plant_and_poi(plant, poi):

    # plant poi 数据合并
    plant_and_poi = plant_and_poi_merge(plant, poi)
    # 缺失值处理
    plant_and_poi = plant_and_poi_missing_value_process(plant_and_poi)

    # 特征编码
    plant_and_poi = plant_and_poi_encoding(plant_and_poi)
    # print("---------------nan值处理-----------------")
    # print(plant_and_poi)


    return plant_and_poi

def preprocess_plant_and_poi_desc(plant, poi):

    # plant poi 数据合并
    plant_and_poi = plant_and_poi_merge(plant, poi)
    # 缺失值处理
    plant_and_poi = plant_and_poi_missing_value_process(plant_and_poi)

    # 特征编码
    # plant_and_poi = plant_and_poi_encoding(plant_and_poi)
    # print("---------------nan值处理-----------------")
    # print(plant_and_poi)


    return plant_and_poi


def preprocess_bill_and_plant_desc(bill, plant_desc):

    bill_and_plant_desc = bill_and_plant_desc_merge(bill, plant_desc)

    return bill_and_plant_desc

def preprocess_bill_and_promotion(bill, promotion):
    bill_and_promotion = bill_and_promotion_merge(bill, promotion)

    return bill_and_promotion
