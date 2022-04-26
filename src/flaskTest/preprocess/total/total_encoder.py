from data_encoding.one_hot_encoder import one_hot_encoding


def total_encoding(total):

    # col_set_onehot = ['plant_asset', 'road_class', 'plant_class_code', 'plant_location_class', 'promotion_type', 'plant_type_desc']

    # total = LabelEncoder_asset(total)
    # total = LabelEncoder_roadClass(total)
    # total = LabelEncoder_plantClass(total)
    # total = LabelEncoder_locationClass(total)
    # total = LabelEncoder_promotionType(total)
    # LabelEncoder_plantType(total)

    #total = OneHotEncoder_asset(total)
    #total = OneHotEncoder_roadClass(total)
    #total = OneHotEncoder_plantClass(total)
    #total = OneHotEncoder_locationClass(total)
    total = OneHotEncoder_promotionType(total)
    total = OneHotEncoder_storeClassDesc(total)
    #total = OneHotEncoder_plantType(total)
    total = OneHotEncoder_plant_city_desc(total)


    store = 'store_class'
    plant = 'plant_stars'
    keyanliang = 'plant_keyanliang_desc'

    #total = LabelEncoder_stars(total, store)
    #total = LabelEncoder_stars(total, plant)
    #total = LabelEncoder_keyanliang(total, keyanliang)
    return total

def plant_and_poi_encoding(plant_and_poi):

    # LabelEncoder_asset(plant_and_poi)
    # LabelEncoder_roadClass(plant_and_poi)
    # LabelEncoder_plantClass(plant_and_poi)
    # LabelEncoder_locationClass(plant_and_poi)
    # LabelEncoder_plantType(plant_and_poi)

    plant_and_poi = OneHotEncoder_asset(plant_and_poi)
    plant_and_poi = OneHotEncoder_roadClass(plant_and_poi)
    plant_and_poi = OneHotEncoder_plantClass(plant_and_poi)
    plant_and_poi = OneHotEncoder_locationClass(plant_and_poi)
    plant_and_poi = OneHotEncoder_plantType(plant_and_poi)

    store = 'store_class'
    plant = 'plant_stars'
    keyanliang = 'plant_keyanliang_desc'

    # LabelEncoder_stars(plant_and_poi, store)
    LabelEncoder_stars(plant_and_poi, plant)
    LabelEncoder_keyanliang(plant_and_poi, keyanliang)

    return plant_and_poi

def LabelEncoder_stars(df, col):

    df.loc[df[col] == '11', col] = 0;
    df.loc[df[col] == '10', col] = 1;
    df.loc[df[col] == '01', col] = 2;
    df.loc[df[col] == '02', col] = 3;
    df.loc[df[col] == '03', col] = 4;
    df.loc[df[col] == '04', col] = 5;
    df.loc[df[col] == '05', col] = 6;
    #df[col] = df[col].astype('int')
    df[col] = df[col].astype('float')
    # z-score标准化
    df[col] = (df[col] - df[col].mean()) / (df[col].std())
    return df



def LabelEncoder_keyanliang(df, col):

    df.loc[df[col] == '年销量500-1000T', col] = 1;
    df.loc[df[col] == '年销量1000-3000T', col] = 2;
    df.loc[df[col] == '年销量3000-5000T', col] = 3;
    df.loc[df[col] == '年销量5000-10000T', col] = 4;
    df.loc[df[col] == '年销量10000T以上', col] = 5;
    #df[col] = df[col].astype('int')
    df[col] = df[col].astype('float')
    # z-score标准化
    df[col] = (df[col] - df[col].mean()) / (df[col].std())
    return df


def LabelEncoder_asset(df):
    col = "plant_asset"
    df.loc[df[col] == '0001', col] = 1;
    df.loc[df[col] == '0002', col] = 2;
    df.loc[df[col] == '0003', col] = 3;
    df.loc[df[col] == '0004', col] = 4;
    df[col] = df[col].astype('float')
    # z-score标准化
    df[col] = (df[col] - df[col].mean()) / (df[col].std())
    return df
def OneHotEncoder_asset(df):
    col = 'plant_asset'
    df = one_hot_encoding(df, col)
    df = df.drop(columns=[col])
    return df

def LabelEncoder_roadClass(df):
    col = "road_class"
    df.loc[df[col] == '0001', col] = 4;
    df.loc[df[col] == '0002', col] = 3;
    df.loc[df[col] == '0003', col] = 2;
    df.loc[df[col] == '0004', col] = 1;
    df[col] = df[col].astype('float')
    # z-score标准化
    df[col] = (df[col] - df[col].mean()) / (df[col].std())
    return df
def OneHotEncoder_roadClass(df):
    col = 'road_class'
    df = one_hot_encoding(df, col)
    df = df.drop(columns=[col])
    return df

def LabelEncoder_plantClass(df):
    col = "plant_class_code"
    df.loc[df[col] == '0001', col] = 4;
    df.loc[df[col] == '0002', col] = 3;
    df.loc[df[col] == '0003', col] = 2;
    df.loc[df[col] == '0004', col] = 1;
    df[col] = df[col].astype('float')
    # z-score标准化
    df[col] = (df[col] - df[col].mean()) / (df[col].std())
    return df
def OneHotEncoder_plantClass(df):
    col = 'plant_class_code'
    df = one_hot_encoding(df, col)
    df = df.drop(columns=[col])
    return df

def LabelEncoder_locationClass(df):
    col = 'plant_location_class'
    df.loc[df[col] == '001', col] = 12;
    df.loc[df[col] == '002', col] = 11;
    df.loc[df[col] == '003', col] = 10;
    df.loc[df[col] == '004', col] = 9;
    df.loc[df[col] == '005', col] = 8;
    df.loc[df[col] == '006', col] = 7;
    df.loc[df[col] == '007', col] = 6;
    df.loc[df[col] == '008', col] = 5;
    df.loc[df[col] == '009', col] = 4;
    df.loc[df[col] == '010', col] = 3;
    df.loc[df[col] == '011', col] = 2;
    df.loc[df[col] == '012', col] = 1;
    df[col] = df[col].astype('float')
    # z-score标准化
    df[col] = (df[col] - df[col].mean()) / (df[col].std())
    return df
def OneHotEncoder_locationClass(df):
    col = 'plant_location_class'
    df = one_hot_encoding(df, col)
    df = df.drop(columns=[col])
    return df


def LabelEncoder_promotionType(df):
    col = "promotion_type"
    df.loc[df[col] == 'NP', col] = 1;
    df.loc[df[col] == 'C', col] = 2;
    df.loc[df[col] == 'O', col] = 3;
    df.loc[df[col] == 'A', col] = 4;
    df.loc[df[col] == 'G', col] = 5;
    df.loc[df[col] == 'BB01', col] = 6;
    df.loc[df[col] == 'BB02', col] = 7;
    df.loc[df[col] == 'BB03', col] = 8;
    df.loc[df[col] == 'BB04', col] = 9;
    df.loc[df[col] == 'BB05', col] = 10;
    df.loc[df[col] == 'BB06', col] = 11;
    df.loc[df[col] == 'BB07', col] = 12;
    df.loc[df[col] == 'BB08', col] = 13;
    df.loc[df[col] == 'BB09', col] = 14;
    df.loc[df[col] == 'BB10', col] = 15;
    df.loc[df[col] == 'BB11', col] = 16;
    df.loc[df[col] == 'BB14', col] = 17;
    df.loc[df[col] == 'P', col] = 18;
    df.loc[df[col] == 'BB31', col] = 19;
    df.loc[df[col] == 'BB20', col] = 20;
    df.loc[df[col] == 'BB24', col] = 24;
    df[col] = df[col].astype('float')
    # z-score标准化
    df[col] = (df[col] - df[col].mean()) / (df[col].std())
    return df
def OneHotEncoder_promotionType(df):
    col = 'promotion_type'
    df = one_hot_encoding(df, col)
    df = df.drop(columns=[col])
    return df

def OneHotEncoder_storeClassDesc(df):
    col = 'store_class_desc'
    df = one_hot_encoding(df, col)
    df = df.drop(columns=[col])
    return df

def LabelEncoder_plantType(df):
    col = "plant_type_desc"
    df.loc[df[col] == '普通站', col] = 1;
    df.loc[df[col] == '纯便利店', col] = 2;
    df.loc[df[col] == 'D类站', col] = 3;
    df.loc[df[col] == '中转仓', col] = 4;
    df.loc[df[col] == '虚拟站', col] = 5;
    df.loc[df[col] == '中央仓', col] = 6;
    df[col] = df[col].astype('float')
    # z-score标准化
    # df[col] = (df[col] - df[col].mean()) / (df[col].std())
    # return df
def OneHotEncoder_plantType(df):
    col = 'plant_type_desc'
    df = one_hot_encoding(df, col)
    df = df.drop(columns=[col])
    return df

def OneHotEncoder_plant_city_desc(df):
    col = 'plant_city_desc'
    df = one_hot_encoding(df, col)
    df = df.drop(columns=[col])
    return df





