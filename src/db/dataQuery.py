from db.dateBase import *
import time


def getMaterialBigcategory(material):
    db = getDBConnect()

    sql = "SELECT DISTINCT big_category FROM table_material_category " \
          "where material = '" + material + "';"
    cursor = db.cursor()
    cursor.execute(sql)
    db.commit()
    cursor.close()
    db.close()

    df_category = cursor.fetchall()
    # 获取连接对象的描述信息
    columnDes = cursor.description
    columnNames = [columnDes[i][0] for i in range(len(columnDes))]
    df_category = pd.DataFrame([list(i) for i in df_category], columns=columnNames)
    df_category = df_category['big_category']
    big_category = df_category[0]
    return big_category

def getPlantsId():
    db = getDBConnect()

    sql = "SELECT DISTINCT plant FROM table_plant_analysis"
    cursor = db.cursor()
    cursor.execute(sql)
    db.commit()
    cursor.close()
    db.close()

    df_plants = cursor.fetchall()
    # 获取连接对象的描述信息
    columnDes = cursor.description
    columnNames = [columnDes[i][0] for i in range(len(columnDes))]
    df_plants = pd.DataFrame([list(i) for i in df_plants], columns=columnNames)
    return df_plants.plant


def getMaterialsId():
    db = getDBConnect()

    # sql = "SELECT distinct  material from material_on_sale"
    #sql = "SELECT distinct  material from material_on_sale WHERE material not in (SELECT DISTINCT material FROM pearson)"
    sql = "SELECT DISTINCT a.material FROM material_on_sale a join table_material_sales_num_record b on a.material = b.material WHERE b.record >= 50"
    cursor = db.cursor()
    cursor.execute(sql)
    cursor.close()
    db.commit()
    db.close()

    df_materials = cursor.fetchall()
    # 获取连接对象的描述信息
    columnDes = cursor.description
    columnNames = [columnDes[i][0] for i in range(len(columnDes))]
    df_materials = pd.DataFrame([list(i) for i in df_materials], columns=columnNames)

    return df_materials.material

# 通过商品编号获取商品销售影响因素的相关系数
def getMaterialsPearson(material):
    db = getDBConnect()

    sql = "select material, promotion_type, discount_rate2, number_station, number_store, " \
          "plant_asset, road_class, plant_stars, store_class, building_area, business_hall, " \
          "paking_area, store_area, plant_class_code, plant_location_class, plant_keyanliang_desc, " \
          "plant_type_desc, type01, type02, type03, type05, type06, type07, type08, type10, type11,  " \
          "type12, type13, type14,type15, type16, type17, type19, type20, type97 " \
          "from pearson_type " \
          "WHERE material = '" + material + "'"
    cursor = db.cursor()
    cursor.execute(sql)
    cursor.close()
    db.commit()
    db.close()

    df_pearson = cursor.fetchall()
    # 获取连接对象的描述信息
    columnDes = cursor.description
    columnNames = [columnDes[i][0] for i in range(len(columnDes))]
    df_pearson = pd.DataFrame([list(i) for i in df_pearson], columns=columnNames)

    return df_pearson

# 通过大类编码获取按类别的pearson相关系数
def getMaterialsPearsonByBigCategory(bigCategory):
    db = getDBConnect()

    sql = "select big_category, promotion_type, discount_rate2, number_station, number_store, " \
          "plant_asset, road_class, plant_stars, store_class, building_area, business_hall, " \
          "paking_area, store_area, plant_class_code, plant_location_class, plant_keyanliang_desc, " \
          "type01, type02, type03, type05, type06, type07, type08, type10, type11,  " \
          "type12, type13, type14,type15, type16, type17, type19, type20, type97 " \
          "from table_pearson_groupby_category3 " \
          "WHERE big_category = '" + bigCategory + "'"
    cursor = db.cursor()
    cursor.execute(sql)
    cursor.close()
    db.commit()
    db.close()

    df_pearson = cursor.fetchall()
    # 获取连接对象的描述信息
    columnDes = cursor.description
    columnNames = [columnDes[i][0] for i in range(len(columnDes))]
    df_pearson = pd.DataFrame([list(i) for i in df_pearson], columns=columnNames)

    return df_pearson

# 通过商品编码获取回归分析结果
def getMaterialsRegression(material):
    db = getDBConnect()

    sql = "select material, promotion_type, discount_rate2, number_station, number_store, " \
          "plant_asset, road_class, plant_stars, store_class, building_area, business_hall, " \
          "paking_area, store_area, plant_class_code, plant_location_class, plant_keyanliang_desc, " \
          " type01, type02, type03, type05, type06, type07, type08, type10, type11,  " \
          "type12, type13, type14,type15, type16, type17, type19, type20, type97 , r " \
          "from regression_type " \
          "WHERE material = '" + material + "'"
    cursor = db.cursor()
    cursor.execute(sql)
    cursor.close()
    db.commit()
    db.close()

    df_regression = cursor.fetchall()
    # 获取连接对象的描述信息
    columnDes = cursor.description
    columnNames = [columnDes[i][0] for i in range(len(columnDes))]
    df_regression = pd.DataFrame([list(i) for i in df_regression], columns=columnNames)

    r = df_regression['r'].iloc[0]
    df_regression = df_regression.drop(labels=None, axis=1, index=None, columns=['material','r'], inplace=False)

    return df_regression, r

# 通过商品编码获取商品名称
def getMaterialName(material):


    db = getDBConnect()
    sql = " SELECT materialtxt FROM material WHERE material = '" + material + "'"
    cursor = db.cursor()
    cursor.execute(sql)
    cursor.close()
    db.commit()
    db.close()

    df_name = cursor.fetchall()
    # 获取连接对象的描述信息
    columnDes = cursor.description
    columnNames = [columnDes[i][0] for i in range(len(columnDes))]
    df_name = pd.DataFrame([list(i) for i in df_name], columns=columnNames)
    df_name = df_name['materialtxt']
    name = df_name[0]

    return name

# 通过大类编码获取大类名称
def getBigCategoryName(bigCategory):

    db = getDBConnect()
    sql = " SELECT distinct big_category_name FROM table_materialOnSale_category WHERE big_category = '" + bigCategory + "'"
    cursor = db.cursor()
    cursor.execute(sql)
    cursor.close()
    db.commit()
    db.close()

    df_name = cursor.fetchall()
    # 获取连接对象的描述信息
    columnDes = cursor.description
    columnNames = [columnDes[i][0] for i in range(len(columnDes))]
    df_name = pd.DataFrame([list(i) for i in df_name], columns=columnNames)
    df_name = df_name['big_category_name']
    name = df_name[0]

    return name



#print(getMaterialName('000000000055001060'))

# print(getPlantsId())

#print(getBigCategoryName('2018'))
#print(getMaterialBigcategory('000000000055001060'))