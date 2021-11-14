from db.dataQuery import getMaterialBigcategory
from db.dateBase import *
import numpy as np



# def insert_table_total(material , df):
#     db = getDBConnect()
#     big_category = getMaterialBigcategory(material)
#
#     try:
#         for i in range(0,len(df)):
#             count = 0;
#             sql = "insert into table_total (material, plant, date, quantity, promotion_type, discount_rate2,number_station, number_store, " \
#                   "plant_asset, road_class,plant_stars, store_class, building_area, business_hall,paking_area, store_area, " \
#                   "plant_class_code, plant_location_class,plant_keyanliang_desc, plant_type_desc, type05, type06,type07, type10," \
#                   "type12, type14, type17, type11, type08,type03, type20, type16, type01, type19, type15, type97,type02, type13, big_category) " \
#                   "VALUES('%s', '%s', '%f','%f', '%d', '%f', '%f', '%f', '%f', '%f', '%f', '%f', '%f', '%f', '%f', '%f', '%f', '%f'," \
#                   " '%f', '%f', '%f', '%f', '%f', '%f', '%f', '%f', '%f', '%f', '%f', '%f', '%f', '%f', '%f', '%f', '%f', '%f', '%f','%f', '%s');" \
#                   % (material, df.iloc[i]['plant'], df.iloc[i]['date'] , df.iloc[i]['quantity'] , df.iloc[i]['promotion_type'], df.iloc[i]['discount_rate2'],
#                      df.iloc[i]['number_station'], df.iloc[i]['number_store'], df.iloc[i]['plant_asset'], df.iloc[i]['road_class'],df.iloc[i]['plant_stars'],
#                      df.iloc[i]['store_class'], df.iloc[i]['building_area'], df.iloc[i]['business_hall'], df.iloc[i]['paking_area'], df.iloc[i]['store_area'],
#                      df.iloc[i]['plant_class_code'], df.iloc[i]['plant_location_class'],df.iloc[i]['plant_keyanliang_desc'],df.iloc[i]['plant_type_desc'],
#                      df.iloc[i]['type05'], df.iloc[i]['type06'], df.iloc[i]['type07'], df.iloc[i]['type10'], df.iloc[i]['type12'], df.iloc[i]['type14'],
#                      df.iloc[i]['type17'], df.iloc[i]['type11'],df.iloc[i]['type08'], df.iloc[i]['type03'], df.iloc[i]['type20'], df.iloc[i]['type16'],
#                      df.iloc[i]['type01'], df.iloc[i]['type19'], df.iloc[i]['type15'], df.iloc[i]['type97'], df.iloc[i]['type02'], df.iloc[i]['type13'], big_category)
#             cursor = db.cursor()
#             cursor.execute(sql)
#             db.commit()
#             count = count + 1;
#             print("count : " + count)
#             cursor.close()
#         print("[info : database] " + material + "total info insert ok!")
#     except Exception as err:
#         print('An exception happend at material: ' + material)
#         print('An exception happend: ' + str(err))
#         db.rollback()
#     finally:
#             db.close()

def insert_table_total(material , df):
    db = getDBConnect()
    big_category = getMaterialBigcategory(material)


    for i in range(0,len(df)):
        count = 0;
        sql = "insert into table_total (material, plant, date, quantity, promotion_type, discount_rate2,number_station, number_store, " \
                  "plant_asset, road_class,plant_stars, store_class, building_area, business_hall,paking_area, store_area, " \
                  "plant_class_code, plant_location_class,plant_keyanliang_desc, plant_type_desc, type05, type06,type07, type10," \
                  "type12, type14, type17, type11, type08,type03, type20, type16, type01, type19, type15, type97,type02, type13, big_category) " \
                  "VALUES('%s', '%s', '%f','%f', '%d', '%f', '%f', '%f', '%f', '%f', '%f', '%f', '%f', '%f', '%f', '%f', '%f', '%f'," \
                  " '%f', '%f', '%f', '%f', '%f', '%f', '%f', '%f', '%f', '%f', '%f', '%f', '%f', '%f', '%f', '%f', '%f', '%f', '%f','%f', '%s');" \
                  % (material, df.iloc[i]['plant'], df.iloc[i]['date'] , df.iloc[i]['quantity'] , df.iloc[i]['promotion_type'], df.iloc[i]['discount_rate2'],
                     df.iloc[i]['number_station'], df.iloc[i]['number_store'], df.iloc[i]['plant_asset'], df.iloc[i]['road_class'],df.iloc[i]['plant_stars'],
                     df.iloc[i]['store_class'], df.iloc[i]['building_area'], df.iloc[i]['business_hall'], df.iloc[i]['paking_area'], df.iloc[i]['store_area'],
                     df.iloc[i]['plant_class_code'], df.iloc[i]['plant_location_class'],df.iloc[i]['plant_keyanliang_desc'],df.iloc[i]['plant_type_desc'],
                     df.iloc[i]['type05'], df.iloc[i]['type06'], df.iloc[i]['type07'], df.iloc[i]['type10'], df.iloc[i]['type12'], df.iloc[i]['type14'],
                     df.iloc[i]['type17'], df.iloc[i]['type11'],df.iloc[i]['type08'], df.iloc[i]['type03'], df.iloc[i]['type20'], df.iloc[i]['type16'],
                     df.iloc[i]['type01'], df.iloc[i]['type19'], df.iloc[i]['type15'], df.iloc[i]['type97'], df.iloc[i]['type02'], df.iloc[i]['type13'], big_category)
        cursor = db.cursor()
        cursor.execute(sql)
        db.commit()
        count = count + 1;
        print("count : %d" % count)
        cursor.close()
    print("[info : database] " + material + "total info insert ok!")
    db.close()







def insert_table_pearson(material,df):

    db = getDBConnect()

    print('pearson:---------')
    print(df)
    print('----------')
    pearson = df.quantity
    print(pearson)

    for i in range(len(pearson)):
        if (np.isnan(pearson[i])):
            pearson[i] = 0

    print("pearson promotion_type ----------------")

    # 销量影响因素所计算的pearson系数
    sql = "INSERT into pearson_type2(material,promotion_type,discount_rate2,number_station,number_store,plant_asset,road_class," \
          "plant_stars,store_class,building_area,business_hall,paking_area,store_area,plant_class_code,plant_location_class," \
          "plant_keyanliang_desc, type05 , type06, type07, type10, type12, type14, type17, type11, type08, type03, type20, type16, type01, type19, type15, type97, type02, type13) " \
          "VALUES('%s','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f');" % \
          (material, pearson['promotion_type'], pearson['discount_rate2'], pearson['number_station'], pearson['number_store'], pearson['plant_asset'], pearson['road_class'],
           pearson['plant_stars'], pearson['store_class'], pearson['building_area'], pearson['business_hall'], pearson['paking_area'], pearson['store_area'], pearson['plant_class_code'], pearson['plant_location_class'],
           pearson['plant_keyanliang_desc'], pearson['type05'], pearson['type06'], pearson['type07'], pearson['type10'], pearson['type12'], pearson['type14'], pearson['type17'], pearson['type11'],
           pearson['type08'], pearson['type03'], pearson['type20'], pearson['type16'], pearson['type01'], pearson['type19'], pearson['type15'], pearson['type97'], pearson['type02'], pearson['type13'])
    # 进站人数影响因素所计算的pearson系数
    # sql = "INSERT into pearson_number_store(material,promotion_type,discount_rate2,number_station,number_store,plant_asset,road_class," \
    #       "plant_stars,store_class,building_area,business_hall,paking_area,store_area,plant_class_code,plant_location_class," \
    #       "plant_keyanliang_desc, type05 , type06, type07, type10, type12, type14, type17, type11, type08, type03, type20, type16, type01, type19, type15, type97, type02, type13) " \
    #       "VALUES('%s','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f');" % \
    #       (material, pearson[1], pearson[2], pearson[3], pearson[4], pearson[5], pearson[6], pearson[7], pearson[8],
    #        pearson[9], pearson[10],
    #        pearson[11], pearson[12], pearson[13], pearson[14], pearson[15], pearson[16], pearson[17], pearson[18],
    #        pearson[19], pearson[20], pearson[21]
    #        , pearson[22], pearson[23], pearson[24], pearson[25], pearson[26], pearson[27], pearson[28], pearson[29],
    #        pearson[30], pearson[31], pearson[32], pearson[33])

    cursor = db.cursor()
    cursor.execute(sql)
    cursor.close()
    db.commit()
    db.close()


def insert_table_regression(material,a,b,r):

    db = getDBConnect()



    # 销量影响因素
    sql = "INSERT into regression_type2(material,jieju,promotion_type,discount_rate2,number_station,number_store,plant_asset,road_class," \
          "plant_stars,store_class,building_area,business_hall,paking_area,store_area,plant_class_code,plant_location_class," \
          "plant_keyanliang_desc,r, type05 , type06, type07, type10, type12, type14, type17, type11, type08, type03, type20, type16, type01, type19, type15, type97, type02, type13) " \
          "VALUES('%s','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f');" % \
          (material, float(a),float(b[0]),float(b[1]),float(b[2]), float(b[3]), float(b[4]), float(b[5]),float(b[6]), float(b[7]), float(b[8]),float(b[9]), float(b[10]),
           float(b[11]), float(b[12]), float(b[13]), float(b[14]), float(r),float(b[15]),float(b[16]),float(b[17]),float(b[18]),float(b[19]),float(b[20]),float(b[21]),float(b[22]),
           float(b[23]),float(b[24]),float(b[25]),float(b[26]),float(b[27]),float(b[28]),float(b[29]),float(b[30]),float(b[31]),float(b[32]))

    # 进店人数影响因素
    # sql = "INSERT into regression_type(material,jieju,promotion_type,discount_rate2,number_station,plant_asset,road_class," \
    #       "plant_stars,store_class,building_area,business_hall,paking_area,store_area,plant_class_code,plant_location_class," \
    #       "plant_keyanliang_desc,r, type05 , type06, type07, type10, type12, type14, type17, type11, type08, type03, type20, type16, type01, type19, type15, type97, type02, type13) " \
    #       "VALUES('%s','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f');" % \
    #       (
    #       material, float(a), float(b[0]), float(b[1]), float(b[2]), float(b[3]), float(b[4]), float(b[5]), float(b[6]),
    #       float(b[7]), float(b[8]), float(b[9]), float(b[10]),
    #       float(b[11]), float(b[12]), float(b[13]), float(b[14]), float(r), float(b[15]), float(b[16]), float(b[17]),
    #       float(b[18]), float(b[19]), float(b[20]), float(b[21]), float(b[22]),
    #       float(b[23]), float(b[24]), float(b[25]), float(b[26]), float(b[27]), float(b[28]), float(b[29]),
    #       float(b[30]), float(b[31]), float(b[32]))

    cursor = db.cursor()
    cursor.execute(sql)
    cursor.close()
    db.commit()
    db.close()


def insert_table_material(materials):

    db = getDBConnect()
    for i in range(len(materials)):
        sql = "INSERT into material_on_sale(material) VALUES('%s');" % (materials[i])
        cursor = db.cursor()
        cursor.execute(sql)
        cursor.close()
    db.commit()
    db.close()




