from flaskTest.db.bill import *
from flaskTest.preprocess.bill.bill_preprocess import preprocess_bill


def get_bill_by_material(material, start_time, end_time):

    bill = get_bill_from_db(material, start_time, end_time)
    bill = preprocess_bill(bill)


    # print(bill.info)
    # print(bill.describe().T)
    # print(bill.dtypes)

    return bill


def get_bill_by_category(category, start_time, end_time):
    bill = get_bill_by_category_from_db(category, start_time, end_time)
    bill = preprocess_bill(bill)

    # print(bill.info)
    # print(bill.describe().T)
    # print(bill.dtypes)

    return bill


# 查询商品在指定分公司的销售数据
def get_bill_by_material_and_company(material, start_time, end_time, company):

    bill = get_bill_from_db_by_material_and_company(material, start_time, end_time, company)
    bill = preprocess_bill(bill)


    # print(bill.info)
    # print(bill.describe().T)
    # print(bill.dtypes)

    return bill


def get_bill_by_material_and_plant_class(material, start_time, end_time, plant_class):
    bill = get_bill_from_db_by_material_and_plant_class(material, start_time, end_time, plant_class)
    bill = preprocess_bill(bill)

    # print(bill.info)
    # print(bill.describe().T)
    # print(bill.dtypes)

    return bill

def get_bill_by_all_material_and_plant_class(start_time, end_time, plant_class):

    bill = get_bill_from_db_by_all_material_and_plant_class(start_time, end_time, plant_class)
    bill = preprocess_bill(bill)

    # print(bill.info)
    # print(bill.describe().T)
    # print(bill.dtypes)

    return bill


def get_bill_by_category_and_company(category, start_time, end_time, company):
    bill = get_bill_from_db_by_category_and_company(category, start_time, end_time, company)
    bill = preprocess_bill(bill)

    # print(bill.info)
    # print(bill.describe().T)
    # print(bill.dtypes)

    return bill


def get_bill_by_category_and_keyanliang(category, start_time, end_time, keyanliang):
    bill = get_bill_from_db_by_category_and_keyanliang(category, start_time, end_time, keyanliang)
    bill = preprocess_bill(bill)

    # print(bill.info)
    # print(bill.describe().T)
    # print(bill.dtypes)

    return bill

def get_bill_by_category_and_stars(category, start_time, end_time, stars):

    bill = get_bill_from_db_by_category_and_stars(category, start_time, end_time, stars)
    bill = preprocess_bill(bill)

    # print(bill.info)
    # print(bill.describe().T)
    # print(bill.dtypes)

    return bill



def get_bill_by_material_and_keyanliang(material, start_time, end_time, keyanliang):

    bill = get_bill_from_db_by_material_and_keyanliang(material, start_time, end_time, keyanliang)

    bill = preprocess_bill(bill)


    # print(bill.info)
    # print(bill.describe().T)
    # print(bill.dtypes)

    return bill

def get_bill_by_material_and_stars(material, start_time, end_time, stars):

    bill = get_bill_from_db_by_material_and_stars(material, start_time, end_time, stars)

    bill = preprocess_bill(bill)

    # print(bill.info)
    # print(bill.describe().T)
    # print(bill.dtypes)

    return bill

# get_bill_by_material('000000000070251989')