

#删除面积字段中多余字符串
def plant_area_m2_delete(plant):
    plant_area_m2_replace(plant, 'building_area')
    plant_area_m2_replace(plant, 'business_hall')
    plant_area_m2_replace(plant, 'paking_area')
    plant_area_m2_replace(plant, 'store_area')

    return plant

#删除df中col字段中的 m2、M2、平米 字符串
def plant_area_m2_replace(df, col):
    df[col].replace('m2', '', regex=True, inplace=True)
    df[col].replace('M2', '', regex=True, inplace=True)
    df[col].replace('平米', '', regex=True, inplace=True)