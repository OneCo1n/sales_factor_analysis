

def plant_dtypes_exg(plant):


    plant['plant'] = plant['plant'].astype('str')
    plant['date'] = plant['date'].astype('datetime64')
    # plant['date'] = plant['date'].astype('str')

    plant['number_station'] = plant['number_station'].astype('int')
    plant['number_store'] = plant['number_store'].astype('int')

    plant['plant_asset'] = plant['plant_asset'].astype('str')
    plant['road_class'] = plant['road_class'].astype('str')
    plant['plant_stars'] = plant['plant_stars'].astype('str')
    plant['store_class'] = plant['store_class'].astype('str')


    plant['building_area'] = plant['building_area'].astype('float64')
    plant['business_hall'] = plant['business_hall'].astype('float64')
    plant['paking_area'] = plant['paking_area'].astype('float64')
    plant['store_area'] = plant['store_area'].astype('float64')

    plant['plant_class_code'] = plant['plant_class_code'].astype('str')
    plant['plant_location_class'] = plant['plant_location_class'].astype('str')
    plant['plant_keyanliang_desc'] = plant['plant_keyanliang_desc'].astype('str')
    plant['plant_type_desc'] = plant['plant_type_desc'].astype('str')

    return plant