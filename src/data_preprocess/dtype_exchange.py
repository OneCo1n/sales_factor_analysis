


def date_conversion_toStr(df):

    for i in df.columns[1:]:
        df[i] = df[i].astype('str')

    return df


def pearsonDytypeExgPlantPOI(df):

    print(df.dtypes)
    #df['material'] = df['material'].astype('str')
    df['plant'] = df['plant'].astype('str')
    df['number_station'] = df['number_station'].astype('float64')
    df['number_store'] = df['number_store'].astype('float64')
    df['plant_asset'] = df['plant_asset'].astype('float64')
    df['road_class'] = df['road_class'].astype('float64')
    df['plant_stars'] = df['plant_stars'].astype('float64')
    df['store_class'] = df['store_class'].astype('float64')
    df['business_hall'] = df['business_hall'].astype('float64')
    df['paking_area'] = df['paking_area'].astype('float64')
    df['store_area'] = df['store_area'].astype('float64')
    df['plant_class_code'] = df['plant_class_code'].astype('float64')
    df['plant_location_class'] = df['plant_location_class'].astype('float64')
    df['plant_keyanliang_desc'] = df['plant_keyanliang_desc'].astype('float64')
    df['type05'] = df['type05'].astype('float64')
    df['type06'] = df['type06'].astype('float64')
    df['type07'] = df['type07'].astype('float64')
    df['type10'] = df['type10'].astype('float64')
    df['type12'] = df['type12'].astype('float64')
    df['type14'] = df['type14'].astype('float64')
    df['type17'] = df['type17'].astype('float64')
    df['type11'] = df['type11'].astype('float64')
    df['type08'] = df['type08'].astype('float64')
    df['type03'] = df['type03'].astype('float64')
    df['type20'] = df['type20'].astype('float64')
    df['type16'] = df['type16'].astype('float64')
    df['type01'] = df['type01'].astype('float64')
    df['type19'] = df['type19'].astype('float64')
    df['type15'] = df['type15'].astype('float64')
    df['type97'] = df['type97'].astype('float64')
    df['type02'] = df['type02'].astype('float64')
    df['type13'] = df['type13'].astype('float64')
    return df

def pearsonDytypeExg(df):

    print(df.dtypes)
    #df['material'] = df['material'].astype('str')
    df['plant'] = df['plant'].astype('str')


    df['quantity'] = df['quantity'].astype('int')
    df['promotion_type'] = df['promotion_type'].astype('int')
    df['discount_rate2'] = df['discount_rate2'].astype('float64')
    df['number_station'] = df['number_station'].astype('float64')
    df['number_store'] = df['number_store'].astype('float64')
    df['plant_asset'] = df['plant_asset'].astype('float64')
    df['road_class'] = df['road_class'].astype('float64')
    df['plant_stars'] = df['plant_stars'].astype('float64')
    df['store_class'] = df['store_class'].astype('float64')
    df['business_hall'] = df['business_hall'].astype('float64')
    df['paking_area'] = df['paking_area'].astype('float64')
    df['store_area'] = df['store_area'].astype('float64')
    df['plant_class_code'] = df['plant_class_code'].astype('float64')
    df['plant_location_class'] = df['plant_location_class'].astype('float64')
    df['plant_keyanliang_desc'] = df['plant_keyanliang_desc'].astype('float64')
    df['plant_type_desc'] = df['plant_type_desc'].astype('float64')
    return df


def data_conversion_toInt(df):
    for i in df.columns[2:]:
        if(i != 'plant_keyanliang_desc'):
            df[i] = df[i].astype('int')

    return df