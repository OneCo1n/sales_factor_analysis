import pandas as pd

def total_data_merge(bill, discount, plant, poi, promotion):

    df_bill_join_promotion = pd.merge(bill, promotion, on=['plant', 'date'], how='left')
    df_plant_join_poi = pd.merge(plant, poi, on=['plant'])
    df_total = pd.merge(df_bill_join_promotion, df_plant_join_poi, on=['plant', 'date'], how='left')

    # print("------------------df_bill_join_promotion----------------------")
    # print(df_bill_join_promotion)
    # print("------------------df_plant_join_poi----------------------")
    # print(df_plant_join_poi)

    return df_total

def total_by_all_material_and_plant_class_data_merge(bill, discount, plant, poi, promotion):
    df_bill_join_promotion = pd.merge(bill, promotion, on=['material', 'plant', 'date'], how='left')
    df_plant_join_poi = pd.merge(plant, poi, on=['plant'])
    df_total = pd.merge(df_bill_join_promotion, df_plant_join_poi, on=['plant', 'date'], how='left')

    # print("------------------df_bill_join_promotion----------------------")
    # print(df_bill_join_promotion)
    # print("------------------df_plant_join_poi----------------------")
    # print(df_plant_join_poi)

    return df_total

def total_category_data_merge(bill, discount, plant, poi, promotion):
    df_bill_join_promotion = pd.merge(bill, promotion, on=['material', 'plant', 'date'], how='left')
    df_plant_join_poi = pd.merge(plant, poi, on=['plant'])
    df_total = pd.merge(df_bill_join_promotion, df_plant_join_poi, on=['plant', 'date'], how='left')

    # print("------------------df_bill_join_promotion----------------------")
    # print(df_bill_join_promotion)
    # print("------------------df_plant_join_poi----------------------")
    # print(df_plant_join_poi)

    return df_total


def plant_and_poi_merge(plant, poi):

    df_plant_and_poi = pd.merge(plant, poi, on=['plant'], how='left')

    print("-----------------df_plant_and_poi---------------------")
    print(df_plant_and_poi)

    return df_plant_and_poi


def bill_and_plant_desc_merge(bill, plant_desc):

    df_bill_and_plant_desc = pd.merge(bill, plant_desc, on=['plant'], how='left')

    print("-----------------df_bill_and_plant-------------------")
    print(df_bill_and_plant_desc)

    return df_bill_and_plant_desc

def bill_and_promotion_merge(bill, promotion):
    df_bill_and_promotion = pd.merge(bill, promotion, on=['plant', 'date'], how='left')

    print("-----------------df_bill_and_promotion-------------------")
    print(df_bill_and_promotion)

    return df_bill_and_promotion
