import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import csv
from pandas.core.frame import DataFrame


def minMaxScaler(df):
    col = df.columns
    df["number_station"] = (df["number_station"] - df["number_station"].min()) / (df["number_station"].max() - df["number_station"].min())
    df["number_store"] = (df["number_store"] - df["number_store"].min()) / (df["number_store"].max() - df["number_store"].min())
    df["plant_asset"] = (df["plant_asset"] - df["plant_asset"].min()) / (
                df["plant_asset"].max() - df["plant_asset"].min())
    df["road_class"] = (df["road_class"] - df["road_class"].min()) / (
                df["road_class"].max() - df["road_class"].min())
    df["plant_stars"] = (df["plant_stars"] - df["plant_stars"].min()) / (df["plant_stars"].max() - df["plant_stars"].min())
    df["store_class"] = (df["store_class"] - df["store_class"].min()) / (df["store_class"].max() - df["store_class"].min())
    df["building_area"] = (df["building_area"] - df["building_area"].min()) / (df["building_area"].max() - df["building_area"].min())
    df["business_hall"] = (df["business_hall"] - df["business_hall"].min()) / (df["business_hall"].max() - df["business_hall"].min())
    df["paking_area"] = (df["paking_area"] - df["paking_area"].min()) / (df["paking_area"].max() - df["paking_area"].min())
    df["store_area"] = (df["store_area"] - df["store_area"].min()) / (df["store_area"].max() - df["store_area"].min())
    df["plant_class_code"] = (df["plant_class_code"] - df["plant_class_code"].min()) / (
            df["plant_class_code"].max() - df["plant_class_code"].min())
    df["plant_location_class"] = (df["plant_location_class"] - df["plant_location_class"].min()) / (
            df["plant_location_class"].max() - df["plant_location_class"].min())
    df["plant_keyanliang_desc"] = (df["plant_keyanliang_desc"] - df["plant_keyanliang_desc"].min()) / (df["plant_keyanliang_desc"].max() - df["plant_keyanliang_desc"].min())
    df["plant_type_desc"] = (df["plant_type_desc"] - df["plant_type_desc"].min()) / (
            df["plant_type_desc"].max() - df["plant_type_desc"].min())

    # df = (df - df.min()) / (df.max() - df.min())  # 即简单实现标准化
    # return df









# tmp_lst = []
# with open('F:\\论文\\data\\df_encoding.csv', 'r') as f:
#     reader = csv.reader(f)
#     for row in reader:
#         tmp_lst.append(row)
# df = pd.DataFrame(tmp_lst[1:], columns=tmp_lst[0])
# print(df)
# df = df.drop("plant",axis= 1)
# df = df.drop("date",axis = 1)
#
# minMaxScaler(df)
# print(df)
#
# outputpath='F:\论文\data\\biaozhunhua.csv'
# df.to_csv(outputpath,index=True,header=True)