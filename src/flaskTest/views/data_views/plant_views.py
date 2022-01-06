from flaskTest.db.plant import *
from flaskTest.preprocess.plant.plant_preprocess import *


def get_plant(start_time, end_time):

    # 数据库抽取数据
    plant = get_plant_from_db(start_time, end_time)

    plant = preprocess_plant(plant)


    # print(plant.info)
    # print(plant.describe().T)
    # print(plant.dtypes)

    return plant

def get_plant_desc(start_time, end_time):
    # 数据库抽取数据
    plant = get_plant_desc_from_db(start_time, end_time)

    plant = preprocess_plant(plant)


    # print(plant.info)
    # print(plant.describe().T)
    # print(plant.dtypes)

    return plant

def get_plant_desc_by_company(start_time, end_time, company):
    # 数据库抽取数据
    plant = get_plant_desc_by_company_from_db(start_time, end_time, company)

    plant = preprocess_plant(plant)

    # print(plant.info)
    # print(plant.describe().T)
    # print(plant.dtypes)

    return plant

def get_plant_desc_by_keyanliang(start_time, end_time, keyanliang):
    # 数据库抽取数据

    plant = get_plant_desc_by_keyanliang_from_db(start_time, end_time, keyanliang)

    plant = preprocess_plant(plant)

    # print(plant.info)
    # print(plant.describe().T)
    # print(plant.dtypes)

    return plant

def get_plant_desc_by_stars(start_time, end_time, stars):

    plant = get_plant_desc_by_stars_from_db(start_time, end_time, stars)

    plant = preprocess_plant(plant)

    # print(plant.info)
    # print(plant.describe().T)
    # print(plant.dtypes)

    return plant

def get_plant_by_company(start_time, end_time, company):

    # 数据库抽取数据
    plant = query_plant_from_db_by_company(start_time, end_time, company)

    plant = preprocess_plant(plant)

    print(plant.info)
    print(plant.describe().T)
    print(plant.dtypes)

    return plant

def get_plant_by_keyanliang(start_time, end_time, keyanliang):
    # 数据库抽取数据
    plant = query_plant_from_db_by_keyanliang(start_time, end_time, keyanliang)

    plant = preprocess_plant(plant)

    print(plant.info)
    print(plant.describe().T)
    print(plant.dtypes)

    return plant

def get_plant_by_stars(start_time, end_time, stars):
    # 数据库抽取数据

    plant = query_plant_from_db_by_stars(start_time, end_time, stars)

    plant = preprocess_plant(plant)

    print(plant.info)
    print(plant.describe().T)
    print(plant.dtypes)

    return plant

def get_plantInfoDesc():

    plant_desc = query_plantInfoDesc()
    print(plant_desc.info)
    print(plant_desc.describe().T)
    print(plant_desc.dtypes)

    return plant_desc

def get_plantInfoDesc_by_company(company):

    plant_desc = query_plantInfoDesc_by_company(company)
    print(plant_desc.info)
    print(plant_desc.describe().T)
    print(plant_desc.dtypes)

    return plant_desc

# get_plant()

# get_plantInfoDesc()