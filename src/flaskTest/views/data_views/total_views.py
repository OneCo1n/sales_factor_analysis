from flaskTest.preprocess.total.total_preprocess import preprocess_total, preprocess_plant_and_poi, \
    preprocess_bill_and_plant_desc, preprocess_bill_and_promotion, preprocess_total_desc, preprocess_total_category, \
    preprocess_plant_and_poi_desc, preprocess_total_by_all_material_and_plant_class
from flaskTest.views.data_views.bill_views import get_bill_by_material, get_bill_by_material_and_company, \
    get_bill_by_category, get_bill_by_material_and_keyanliang, get_bill_by_material_and_stars, \
    get_bill_by_category_and_company, get_bill_by_category_and_keyanliang, get_bill_by_category_and_stars, \
    get_bill_by_material_and_plant_class, get_bill_by_all_material_and_plant_class
from flaskTest.views.data_views.discount_views import get_discount_by_material, get_discount_by_category
from flaskTest.views.data_views.plant_views import get_plant, get_plantInfoDesc, get_plant_by_company, \
    get_plantInfoDesc_by_company, get_plant_desc, get_plant_desc_by_company, get_plant_by_keyanliang, \
    get_plant_desc_by_keyanliang, get_plant_by_stars, get_plant_desc_by_stars, get_plant_by_plant_class, \
    get_plant_desc_by_plant_class
from flaskTest.views.data_views.poi_views import get_poi
from flaskTest.views.data_views.promotion_views import get_promotion_by_material, get_promotion_desc_by_material, \
    get_promotion_by_category, get_promotion_desc_by_category, get_promotion_by_all_material_and_plant_class
import seaborn as sns
import matplotlib

import matplotlib.pyplot as plt
# sns.set_style('white',{'font.sans-serif':['simhei','Arial']})
sns.set(style='whitegrid',palette='tab10')
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['font.family'] = 'SimHei'
matplotlib.rcParams['font.size'] = 10
matplotlib.rcParams['axes.unicode_minus']=False


def get_total_by_material(material, start_time, end_time):

    # 获取处理好的各个表数据
    bill = get_bill_by_material(material, start_time, end_time)
    # discount = get_discount_by_material(material , start_time, end_time)
    discount = ""
    plant = get_plant(start_time, end_time)
    poi = get_poi()
    promotion = get_promotion_by_material(material, start_time, end_time)

    # 对多个表数据进行合并，对合并之后的数据进行处理
    total = preprocess_total(bill, discount, plant, poi, promotion)
    # print(total.info)
    # print(total.describe().T)
    # print(total.dtypes)

    return total

# 获取整个类别商品的数据
def get_total_by_category(category, start_time, end_time):
    # 获取处理好的各个表数据

    bill = get_bill_by_category(category, start_time, end_time)
    discount = get_discount_by_category(category, start_time, end_time)
    plant = get_plant(start_time, end_time)
    poi = get_poi()
    promotion = get_promotion_by_category(category, start_time, end_time)

    # 对多个表数据进行合并，对合并之后的数据进行处理
    total = preprocess_total_category(bill, discount, plant, poi, promotion)
    # print(total.info)
    # print(total.describe().T)
    # print(total.dtypes)

    return total


def get_total_by_material_and_company(material, start_time, end_time):

    # 获取处理好的各个表数据
    bill = get_bill_by_material_and_company(material, start_time, end_time)
    discount = get_discount_by_material(material , start_time, end_time)
    plant = get_plant()
    poi = get_poi()
    promotion = get_promotion_by_material(material, start_time, end_time)

    # 对多个表数据进行合并，对合并之后的数据进行处理
    total = preprocess_total(bill, discount, plant, poi, promotion)
    # print(total.info)
    # print(total.describe().T)
    # print(total.dtypes)

    return total

def get_total_desc_by_material(material , start_time, end_time):

    # 获取处理好的各个表数据
    bill = get_bill_by_material(material, start_time, end_time)
    # discount = get_discount_by_material(material, start_time, end_time)
    discount = ""
    plant = get_plant_desc(start_time, end_time)
    poi = get_poi()
    promotion = get_promotion_desc_by_material(material, start_time, end_time)

    # 对多个表数据进行合并，对合并之后的数据进行处理
    total = preprocess_total_desc(bill, discount, plant, poi, promotion)
    # print(total.info)
    # print(total.describe().T)
    # print(total.dtypes)

    return total

def get_total_desc_by_category(category, start_time, end_time):

    # 获取处理好的各个表数据
    bill = get_bill_by_category(category, start_time, end_time)
    discount = ""
    plant = get_plant_desc(start_time, end_time)
    poi = get_poi()
    promotion = get_promotion_desc_by_category(category, start_time, end_time)

    # 对多个表数据进行合并，对合并之后的数据进行处理
    total = preprocess_total_desc(bill, discount, plant, poi, promotion)
    # print(total.info)
    # print(total.describe().T)
    # print(total.dtypes)

    return total

def get_total_desc_by_material_and_company(material, start_time, end_time, company):
    # 获取处理好的各个表数据
    bill = get_bill_by_material_and_company(material, start_time, end_time, company)
    discount = ""
    plant = get_plant_desc_by_company(start_time, end_time, company)
    poi = get_poi()
    promotion = get_promotion_desc_by_material(material, start_time, end_time)

    # 对多个表数据进行合并，对合并之后的数据进行处理
    total = preprocess_total_desc(bill, discount, plant, poi, promotion)
    # print(total.info)
    # print(total.describe().T)
    # print(total.dtypes)

    return total

def get_total_desc_by_material_and_plant_class(material, start_time, end_time, plant_class):
    # 获取处理好的各个表数据
    bill = get_bill_by_material_and_plant_class(material, start_time, end_time, plant_class)
    # discount = get_discount_by_material(material, start_time, end_time)
    discount = ''
    plant = get_plant_desc_by_plant_class(start_time, end_time, plant_class)
    poi = get_poi()
    promotion = get_promotion_desc_by_material(material, start_time, end_time)

    # 对多个表数据进行合并，对合并之后的数据进行处理
    total = preprocess_total_desc(bill, discount, plant, poi, promotion)
    # print(total.info)
    # print(total.describe().T)
    # print(total.dtypes)

    return total

def get_total_desc_by_category_and_company(category, start_time, end_time, company):
    # 获取处理好的各个表数据
    bill = get_bill_by_category_and_company(category, start_time, end_time, company)
    discount = ""
    plant = get_plant_desc_by_company(start_time, end_time, company)
    poi = get_poi()
    promotion = get_promotion_desc_by_category(category, start_time, end_time)

    # 对多个表数据进行合并，对合并之后的数据进行处理
    total = preprocess_total_desc(bill, discount, plant, poi, promotion)
    # print(total.info)
    # print(total.describe().T)
    # print(total.dtypes)

    return total

def get_total_desc_by_category_and_keyanliang(category, start_time, end_time, keyanliang):
    # 获取处理好的各个表数据
    bill = get_bill_by_category_and_keyanliang(category, start_time, end_time, keyanliang)
    discount = ""
    plant = get_plant_desc_by_keyanliang(start_time, end_time, keyanliang)
    poi = get_poi()
    promotion = get_promotion_desc_by_category(category, start_time, end_time)

    # 对多个表数据进行合并，对合并之后的数据进行处理
    total = preprocess_total_desc(bill, discount, plant, poi, promotion)
    # print(total.info)
    # print(total.describe().T)
    # print(total.dtypes)

    return total


def get_total_desc_by_category_and_stars(category, start_time, end_time, stars):
    # 获取处理好的各个表数据
    bill = get_bill_by_category_and_stars(category, start_time, end_time, stars)
    discount = ""
    plant = get_plant_desc_by_stars(start_time, end_time, stars)
    poi = get_poi()
    promotion = get_promotion_desc_by_category(category, start_time, end_time)

    # 对多个表数据进行合并，对合并之后的数据进行处理
    total = preprocess_total_desc(bill, discount, plant, poi, promotion)
    # print(total.info)
    # print(total.describe().T)
    # print(total.dtypes)

    return total



def get_total_desc_by_material_and_keyanliang(material, start_time, end_time, keyanliang):
    # 获取处理好的各个表数据
    bill = get_bill_by_material_and_keyanliang(material, start_time, end_time, keyanliang)
    discount = get_discount_by_material(material, start_time, end_time)
    plant = get_plant_desc_by_keyanliang(start_time, end_time, keyanliang)
    poi = get_poi()
    promotion = get_promotion_desc_by_material(material, start_time, end_time)

    # 对多个表数据进行合并，对合并之后的数据进行处理
    total = preprocess_total_desc(bill, discount, plant, poi, promotion)
    # print(total.info)
    # print(total.describe().T)
    # print(total.dtypes)

    return total

def get_total_desc_by_material_and_stars(material, start_time, end_time, stars):
    # 获取处理好的各个表数据
    bill = get_bill_by_material_and_stars(material, start_time, end_time, stars)
    discount = get_discount_by_material(material, start_time, end_time)
    plant = get_plant_desc_by_stars(start_time, end_time, stars)
    poi = get_poi()
    promotion = get_promotion_desc_by_material(material, start_time, end_time)

    # 对多个表数据进行合并，对合并之后的数据进行处理
    total = preprocess_total_desc(bill, discount, plant, poi, promotion)
    # print(total.info)
    # print(total.describe().T)
    # print(total.dtypes)

    return total



def get_total_by_material_and_company(material,start_time, end_time, company):

    # 获取处理好的各个表数据
    bill = get_bill_by_material_and_company(material, start_time, end_time, company)
    discount = ""
    plant = get_plant_by_company(start_time, end_time, company)
    poi = get_poi()
    promotion = get_promotion_by_material(material, start_time, end_time)

    # 对多个表数据进行合并，对合并之后的数据进行处理
    total = preprocess_total(bill, discount, plant, poi, promotion)
    # print(total.info)
    # print(total.describe().T)
    # print(total.dtypes)

    return total

def get_total_by_material_and_plant_class(material, start_time, end_time, plant_class):
    # 获取处理好的各个表数据
    bill = get_bill_by_material_and_plant_class(material, start_time, end_time, plant_class)

    # discount = get_discount_by_material(material, start_time, end_time)
    discount = ''
    plant = get_plant_by_plant_class(start_time, end_time, plant_class)
    poi = get_poi()
    promotion = get_promotion_by_material(material, start_time, end_time)

    # 对多个表数据进行合并，对合并之后的数据进行处理
    total = preprocess_total(bill, discount, plant, poi, promotion)
    # print(total.info)
    # print(total.describe().T)
    # print(total.dtypes)

    return total

def get_total_by_all_material_and_plant_class(start_time, end_time, plant_class):
    # 获取处理好的各个表数据
    bill = get_bill_by_all_material_and_plant_class(start_time, end_time, plant_class)
    # discount = get_discount_by_material(material, start_time, end_time)
    discount = ''
    plant = get_plant_by_plant_class(start_time, end_time, plant_class)
    poi = get_poi()
    promotion = get_promotion_by_all_material_and_plant_class(start_time, end_time, plant_class)

    # 对多个表数据进行合并，对合并之后的数据进行处理
    total = preprocess_total_by_all_material_and_plant_class(bill, discount, plant, poi, promotion)
    # print(total.info)
    # print(total.describe().T)
    # print(total.dtypes)

    return total

def get_total_by_category_and_company(category, start_time, end_time, company):

    # 获取处理好的各个表数据
    bill = get_bill_by_category_and_company(category, start_time, end_time, company)
    discount = ""
    plant = get_plant_by_company(start_time, end_time, company)
    poi = get_poi()
    promotion = get_promotion_by_category(category, start_time, end_time)

    # 对多个表数据进行合并，对合并之后的数据进行处理
    total = preprocess_total_category(bill, discount, plant, poi, promotion)
    # print(total.info)
    # print(total.describe().T)
    # print(total.dtypes)
    return total

def get_total_by_category_and_keyanliang(category, start_time, end_time, keyanliang):

    # 获取处理好的各个表数据
    bill = get_bill_by_category_and_keyanliang(category, start_time, end_time, keyanliang)
    discount = ""
    plant = get_plant_by_keyanliang(start_time, end_time, keyanliang)
    poi = get_poi()
    promotion = get_promotion_by_category(category, start_time, end_time)

    # 对多个表数据进行合并，对合并之后的数据进行处理
    total = preprocess_total_category(bill, discount, plant, poi, promotion)
    # print(total.info)
    # print(total.describe().T)
    # print(total.dtypes)
    return total

def get_total_by_category_and_stars(category, start_time, end_time, stars):

    # 获取处理好的各个表数据
    bill = get_bill_by_category_and_stars(category, start_time, end_time, stars)
    discount = ""
    plant = get_plant_by_stars(start_time, end_time, stars)
    poi = get_poi()
    promotion = get_promotion_by_category(category, start_time, end_time)

    # 对多个表数据进行合并，对合并之后的数据进行处理
    total = preprocess_total_category(bill, discount, plant, poi, promotion)
    # print(total.info)
    # print(total.describe().T)
    # print(total.dtypes)
    return total


def get_total_by_material_and_keyanliang(material, start_time, end_time, keyanliang):
    # 获取处理好的各个表数据

    bill = get_bill_by_material_and_keyanliang(material, start_time, end_time, keyanliang)
    discount = get_discount_by_material(material, start_time, end_time)
    plant = get_plant_by_keyanliang(start_time, end_time, keyanliang)
    poi = get_poi()
    promotion = get_promotion_by_material(material, start_time, end_time)

    # 对多个表数据进行合并，对合并之后的数据进行处理
    total = preprocess_total(bill, discount, plant, poi, promotion)
    # print(total.info)
    # print(total.describe().T)
    # print(total.dtypes)

    return total

def get_total_by_material_and_stars(material, start_time, end_time, stars):
    # 获取处理好的各个表数据

    bill = get_bill_by_material_and_stars(material, start_time, end_time, stars)
    discount = get_discount_by_material(material, start_time, end_time)
    plant = get_plant_by_stars(start_time, end_time, stars)
    poi = get_poi()
    promotion = get_promotion_by_material(material, start_time, end_time)

    # 对多个表数据进行合并，对合并之后的数据进行处理
    total = preprocess_total(bill, discount, plant, poi, promotion)
    # print(total.info)
    # print(total.describe().T)
    # print(total.dtypes)

    return total



def get_plant_and_poi():

    # 获取处理好的plant和poi信息
    plant = get_plant()
    poi = get_poi()

    # 将plant与poi合并，进行处理
    plant_and_poi = preprocess_plant_and_poi(plant, poi)

    print(plant_and_poi.info)
    print(plant_and_poi.describe().T)
    print(plant_and_poi.dtypes)

    return plant_and_poi

def get_bill_and_plant_desc_by_material(material):

    bill = get_bill_by_material(material)
    plant_desc = get_plantInfoDesc()

    bill_and_plant_desc = preprocess_bill_and_plant_desc(bill, plant_desc)
    print(bill_and_plant_desc.info)
    print(bill_and_plant_desc.describe().T)
    print(bill_and_plant_desc.dtypes)

    return bill_and_plant_desc

def get_bill_and_plant_desc_by_material_and_company(material, company):


    bill = get_bill_by_material_and_company(material, company)
    plant_desc = get_plantInfoDesc_by_company(company)

    bill_and_plant_desc = preprocess_bill_and_plant_desc(bill, plant_desc)
    print(bill_and_plant_desc.info)
    print(bill_and_plant_desc.describe().T)
    print(bill_and_plant_desc.dtypes)

    return bill_and_plant_desc

def get_bill_and_promotion_by_material(material):
    # 获取处理好的各个表数据
    bill = get_bill_by_material(material)
    promotion = get_promotion_desc_by_material(material)

    bill_and_promotion = preprocess_bill_and_promotion(bill, promotion)
    print(bill_and_promotion.info)
    print(bill_and_promotion.describe().T)
    print(bill_and_promotion.dtypes)

    return bill_and_promotion


# 获取用于分析进站进店人数的数据
def get_total_people(start_time, end_time):
    # 获取处理好的plant和poi信息
    plant = get_plant(start_time, end_time)
    poi = get_poi()

    # 将plant与poi合并，进行处理
    plant_and_poi = preprocess_plant_and_poi(plant, poi)
    # 删除无用特征'plant_class_code',
    plant_and_poi = plant_and_poi.drop(labels=None, axis=1, index=None,
                       columns=['building_area', 'business_hall','store_area','store_class',
                                'store_class_desc','plant_stars','plant_keyanliang_desc','number_store',
                                'plant_city_desc'
                                ],
                       inplace=False)

    return plant_and_poi

def get_total_people_by_company(start_time, end_time, company):
    # 获取处理好的plant和poi信息
    plant = get_plant_by_company(start_time, end_time, company)
    poi = get_poi()

    # 将plant与poi合并，进行处理
    plant_and_poi = preprocess_plant_and_poi(plant, poi)
    # 删除无用特征
    plant_and_poi = plant_and_poi.drop(labels=None, axis=1, index=None,
                                       columns=['number_store',],
                                       inplace=False)


    return plant_and_poi

def get_total_people_by_keyanliang(start_time, end_time, keyanliang):
    # 获取处理好的plant和poi信息
    plant = get_plant_by_keyanliang(start_time, end_time, keyanliang)
    poi = get_poi()

    # 将plant与poi合并，进行处理
    plant_and_poi = preprocess_plant_and_poi(plant, poi)
    # 删除无用特征
    plant_and_poi = plant_and_poi.drop(labels=None, axis=1, index=None,
                                       columns=['number_store'],
                                       inplace=False)

    return plant_and_poi

def get_total_people_by_stars(start_time, end_time, stars):
    # 获取处理好的plant和poi信息
    plant = get_plant_by_stars(start_time, end_time, stars)
    poi = get_poi()

    # 将plant与poi合并，进行处理
    plant_and_poi = preprocess_plant_and_poi(plant, poi)
    # 删除无用特征
    plant_and_poi = plant_and_poi.drop(labels=None, axis=1, index=None,
                                       columns=['number_store'],
                                       inplace=False)

    return plant_and_poi


def get_total_people_desc(start_time, end_time):
    # 获取处理好的plant和poi信息
    plant = get_plant_desc(start_time, end_time)
    poi = get_poi()

    # 将plant与poi合并，进行处理
    plant_and_poi = preprocess_plant_and_poi_desc(plant, poi)
    # 删除无用特征
    # plant_and_poi = plant_and_poi.drop(labels=None, axis=1, index=None,
    #                                    columns=['number_store'],
    #                                    inplace=False)

    return plant_and_poi

def get_total_people_desc_by_company(start_time, end_time, company):
    # 获取处理好的plant和poi信息
    plant = get_plant_desc_by_company(start_time, end_time, company)
    poi = get_poi()

    # 将plant与poi合并，进行处理
    plant_and_poi = preprocess_plant_and_poi_desc(plant, poi)
    # 删除无用特征
    # plant_and_poi = plant_and_poi.drop(labels=None, axis=1, index=None,
    #                                    columns=['number_store'],
    #                                    inplace=False)

    return plant_and_poi

def get_total_people_desc_by_keyanliang(start_time, end_time, keyanliang):
    # 获取处理好的plant和poi信息
    plant = get_plant_desc_by_keyanliang(start_time, end_time, keyanliang)
    poi = get_poi()

    # 将plant与poi合并，进行处理
    plant_and_poi = preprocess_plant_and_poi_desc(plant, poi)
    # 删除无用特征
    # plant_and_poi = plant_and_poi.drop(labels=None, axis=1, index=None,
    #                                    columns=['number_store'],
    #                                    inplace=False)

    return plant_and_poi

def get_total_people_desc_by_stars(start_time, end_time, stars):
    # 获取处理好的plant和poi信息
    plant = get_plant_desc_by_stars(start_time, end_time, stars)
    poi = get_poi()

    # 将plant与poi合并，进行处理
    plant_and_poi = preprocess_plant_and_poi_desc(plant, poi)
    # 删除无用特征
    # plant_and_poi = plant_and_poi.drop(labels=None, axis=1, index=None,
    #                                    columns=['number_store'],
    #                                    inplace=False)

    return plant_and_poi







# def get_bill_and_people_by_material(material):
#     bill = get_bill_by_material(material)
#     people = get_all_numberOfPeople()
#
#     bill_and_people = preprocess_bill_and_people(bill, people)
#     print(bill_and_people.info)
#     print(bill_and_people.describe().T)
#     print(bill_and_people.dtypes)
#
#     return bill_and_people



# get_total_by_material('000000000070251989')
# get_plant_and_poi()






