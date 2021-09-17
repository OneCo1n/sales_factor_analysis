from data_encoding.LabelEncoder import *
from data_encoding.one_hot_encoder import *
import time

def encoding(df):

    time_start = time.time()
    print("特征编码开始......")
    #需要热编码的特征
    col_set_onehot = ['plant_asset', 'road_class', 'plant_class_code', 'plant_location_class', 'promotion_type', 'plant_type_desc']
    # df = one_hot_encoding(df, col_set_onehot)
    LabelEncoder_asset(df)
    LabelEncoder_roadClass(df)
    LabelEncoder_plantClass(df)
    LabelEncoder_locationClass(df)
    LabelEncoder_promotionType(df)
    LabelEncoder_plantType(df)

    store = 'store_class'
    plant = 'plant_stars'
    keyanliang = 'plant_keyanliang_desc'

    LabelEncoder_stars(df, store)
    LabelEncoder_stars(df, plant)
    LabelEncoder_keyanliang(df, keyanliang)

    #字段筛选  'plant_asset', 'road_class', 'plant_class_code', 'plant_location_class', 'plant_type_desc','promotion_type'
    df.drop(columns=['discount', 'promotion_quantity', 'promotion_amount', 'retail_price'], inplace=True)
    # 类型转换
    # df_data_plant_join_numofpeople_code['number_station'] = df_data_plant_join_numofpeople_code['number_station'].astype('int')
    for col in df.columns[5:]:
        df[col] = df[col].astype('float64')

    print("特征编码结束 耗时：%.3fs" % (time.time() - time_start))

    return df