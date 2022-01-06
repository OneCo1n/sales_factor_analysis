


def plant_feature_select(plant):

    #plant = plant.drop(labels=None, axis=1, index=None, columns=['plant_type_desc'], inplace=False)
    plant = plant.drop_duplicates(['plant', 'date'])

    return plant