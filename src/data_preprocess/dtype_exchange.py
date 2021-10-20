


def date_conversion_toStr(df):

    for i in df.columns[1:]:
        df[i] = df[i].astype('str')

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
    for i in df.columns[1:]:
        df[i] = df[i].astype('int')

    return df