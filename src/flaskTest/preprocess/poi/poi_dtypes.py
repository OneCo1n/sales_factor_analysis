

def poi_dtypes_exg(poi):

    for i in poi.columns:
        if(i != 'plant'):
            poi[i] = poi[i].astype('int')

    poi['plant'] = poi['plant'].astype('str')

    return poi
