import time

from db.dateBase import *

def get_plant_from_db(start_time, end_time):

    db = getDBConnect()
    cursor = db.cursor()

    sql_select_plant_join_numofpeople = "SELECT a.plant , a.date, a.number_station, a.number_store, " \
                                        "b.plant_asset, b.road_class," \
                                        "b.plant_stars,b.store_class," \
                                        "b.street,b.building_area, b.business_hall," \
                                        "b.paking_area,b.surrounding_building," \
                                        "b.store_area,b.plant_class_code," \
                                        "b.plant_location_class " \
                                        "from table_number_station_store  a left join table_pos_zaplant_xy_orc_508  b " \
                                        "on a.plant = b.bic_zaplant where a.date BETWEEN '2016-01-01' and '2021-06-01' order by a.date"

    sql_select_plant_join_numofpeople_code = "SELECT distinct a.plant , a.date, a.number_station, a.number_store, " \
                                        "b.plant_asset, b.road_class," \
                                        "b.plant_stars,b.store_class," \
                                        "b.building_area, b.business_hall," \
                                        "b.paking_area," \
                                        "b.store_area,b.plant_class_code," \
                                        "b.plant_location_class, b.plant_keyanliang_desc, b.plant_type_desc " \
                                        "from table_number_station_store  a left join table_pos_zaplant_xy_orc_508  b " \
                                        "on a.plant = b.bic_zaplant where a.date BETWEEN '" + start_time + "' and '" + end_time  + "' order by a.date"

    time_sql_start = time.time()
    print("正在获取油站信息......")
    cursor.execute(sql_select_plant_join_numofpeople_code)
    # 使用 fetchall() 方法获取所有数据.以元组形式返回
    data_plant_join_numofpeople_code = cursor.fetchall()
    columnDes = cursor.description
    columnNames = [columnDes[i][0] for i in range(len(columnDes))]
    df_data_plant_join_numofpeople_code = pd.DataFrame([list(i) for i in data_plant_join_numofpeople_code], columns=columnNames)
    time_sql_end = time.time()
    print("已获取油站信息 耗时：%.3fs" % (time_sql_end - time_sql_start))

    #cursor.close()
    closeDBConnect(cursor,db)

    return df_data_plant_join_numofpeople_code

def get_plant_desc_from_db(start_time, end_time):
    db = getDBConnect()
    cursor = db.cursor()

    sql_select_plant_join_numofpeople_code = "SELECT distinct a.plant , a.date, a.number_station, a.number_store, " \
                                             "b.plant_asset_desc as plant_asset, b.road_class_desc as road_class," \
                                             "b.plant_stars_desc as plant_stars,b.store_class_desc as store_class," \
                                             "b.building_area, b.business_hall," \
                                             "b.paking_area," \
                                             "b.store_area,b.plant_class_desc as plant_class_code," \
                                             "b.road_location_desc as plant_location_class, b.plant_keyanliang_desc, b.plant_type_desc " \
                                             " from table_number_station_store  a left join table_pos_zaplant_xy_orc_508  b " \
                                             " on a.plant = b.bic_zaplant where a.date BETWEEN '" + start_time + "' and '" + end_time + "' order by a.date"

    time_sql_start = time.time()
    print("正在获取油站信息......")
    cursor.execute(sql_select_plant_join_numofpeople_code)
    # 使用 fetchall() 方法获取所有数据.以元组形式返回
    data_plant_join_numofpeople_code = cursor.fetchall()
    columnDes = cursor.description
    columnNames = [columnDes[i][0] for i in range(len(columnDes))]
    df_data_plant_join_numofpeople_code = pd.DataFrame([list(i) for i in data_plant_join_numofpeople_code],
                                                       columns=columnNames)
    time_sql_end = time.time()
    print("已获取油站信息 耗时：%.3fs" % (time_sql_end - time_sql_start))

    # cursor.close()
    closeDBConnect(cursor, db)

    return df_data_plant_join_numofpeople_code

def get_plant_desc_by_company_from_db(start_time, end_time, company):
    db = getDBConnect()
    cursor = db.cursor()

    sql_select_plant_join_numofpeople_code = "SELECT distinct a.plant , a.date, a.number_station, a.number_store, " \
                                             "b.plant_asset_desc as plant_asset, b.road_class_desc as road_class," \
                                             "b.plant_stars_desc as plant_stars,b.store_class_desc as store_class," \
                                             "b.building_area, b.business_hall," \
                                             "b.paking_area," \
                                             "b.store_area,b.plant_class_desc as plant_class_code," \
                                             "b.road_location_desc as plant_location_class, b.plant_keyanliang_desc, b.plant_type_desc " \
                                             " from table_number_station_store  a left join table_pos_zaplant_xy_orc_508  b " \
                                             " on a.plant = b.bic_zaplant where a.date BETWEEN '" + start_time + "' and '" + end_time + "'and b.plant_city_desc = '" + company + "' order by a.date "

    time_sql_start = time.time()
    print("正在获取油站信息......")
    cursor.execute(sql_select_plant_join_numofpeople_code)
    # 使用 fetchall() 方法获取所有数据.以元组形式返回
    data_plant_join_numofpeople_code = cursor.fetchall()
    columnDes = cursor.description
    columnNames = [columnDes[i][0] for i in range(len(columnDes))]
    df_data_plant_join_numofpeople_code = pd.DataFrame([list(i) for i in data_plant_join_numofpeople_code],
                                                       columns=columnNames)
    time_sql_end = time.time()
    print("已获取油站信息 耗时：%.3fs" % (time_sql_end - time_sql_start))

    # cursor.close()
    closeDBConnect(cursor, db)

    return df_data_plant_join_numofpeople_code

def get_plant_desc_by_keyanliang_from_db(start_time, end_time, keyanliang):

    db = getDBConnect()
    cursor = db.cursor()

    sql_select_plant_join_numofpeople_code = "SELECT distinct a.plant , a.date, a.number_station, a.number_store, " \
                                             "b.plant_asset_desc as plant_asset, b.road_class_desc as road_class," \
                                             "b.plant_stars_desc as plant_stars,b.store_class_desc as store_class," \
                                             "b.building_area, b.business_hall," \
                                             "b.paking_area," \
                                             "b.store_area,b.plant_class_desc as plant_class_code," \
                                             "b.road_location_desc as plant_location_class, b.plant_keyanliang_desc, b.plant_type_desc " \
                                             " from table_number_station_store  a left join table_pos_zaplant_xy_orc_508  b " \
                                             " on a.plant = b.bic_zaplant where a.date BETWEEN '" + start_time + "' and '" + end_time + "'and b.plant_keyanliang_desc = '" + keyanliang + "' order by a.date "

    time_sql_start = time.time()
    print("正在获取油站信息......")
    cursor.execute(sql_select_plant_join_numofpeople_code)
    # 使用 fetchall() 方法获取所有数据.以元组形式返回
    data_plant_join_numofpeople_code = cursor.fetchall()
    columnDes = cursor.description
    columnNames = [columnDes[i][0] for i in range(len(columnDes))]
    df_data_plant_join_numofpeople_code = pd.DataFrame([list(i) for i in data_plant_join_numofpeople_code],
                                                       columns=columnNames)
    time_sql_end = time.time()
    print("已获取油站信息 耗时：%.3fs" % (time_sql_end - time_sql_start))

    # cursor.close()
    closeDBConnect(cursor, db)

    return df_data_plant_join_numofpeople_code

def get_plant_desc_by_stars_from_db(start_time, end_time, stars):

    db = getDBConnect()
    cursor = db.cursor()

    sql_select_plant_join_numofpeople_code = "SELECT distinct a.plant , a.date, a.number_station, a.number_store, " \
                                             "b.plant_asset_desc as plant_asset, b.road_class_desc as road_class," \
                                             "b.plant_stars_desc as plant_stars,b.store_class_desc as store_class," \
                                             "b.building_area, b.business_hall," \
                                             "b.paking_area," \
                                             "b.store_area,b.plant_class_desc as plant_class_code," \
                                             "b.road_location_desc as plant_location_class, b.plant_keyanliang_desc, b.plant_type_desc " \
                                             " from table_number_station_store  a left join table_pos_zaplant_xy_orc_508  b " \
                                             " on a.plant = b.bic_zaplant where a.date BETWEEN '" + start_time + "' and '" + end_time + "'and b.plant_stars_desc = '" + stars + "' order by a.date "

    time_sql_start = time.time()
    print("正在获取油站信息......")
    cursor.execute(sql_select_plant_join_numofpeople_code)
    # 使用 fetchall() 方法获取所有数据.以元组形式返回
    data_plant_join_numofpeople_code = cursor.fetchall()
    columnDes = cursor.description
    columnNames = [columnDes[i][0] for i in range(len(columnDes))]
    df_data_plant_join_numofpeople_code = pd.DataFrame([list(i) for i in data_plant_join_numofpeople_code],
                                                       columns=columnNames)
    time_sql_end = time.time()
    print("已获取油站信息 耗时：%.3fs" % (time_sql_end - time_sql_start))

    # cursor.close()
    closeDBConnect(cursor, db)

    return df_data_plant_join_numofpeople_code



def query_plant_from_db_by_company(start_time, end_time,company):
    db = getDBConnect()
    cursor = db.cursor()


    sql_select_plant_join_numofpeople_code = "SELECT distinct a.plant , a.date, a.number_station, a.number_store, " \
                                             "b.plant_asset, b.road_class," \
                                             "b.plant_stars,b.store_class," \
                                             "b.building_area, b.business_hall," \
                                             "b.paking_area," \
                                             "b.store_area,b.plant_class_code," \
                                             "b.plant_location_class, b.plant_keyanliang_desc, b.plant_type_desc " \
                                             "from table_number_station_store  a left join table_pos_zaplant_xy_orc_508  b " \
                                             "on a.plant = b.bic_zaplant where a.date BETWEEN '" + start_time + "' and '" + end_time + "'and b.plant_city_desc = '" + company + "'  order by a.date "

    time_sql_start = time.time()
    print("正在获取油站信息......")
    cursor.execute(sql_select_plant_join_numofpeople_code)
    # 使用 fetchall() 方法获取所有数据.以元组形式返回
    data_plant_join_numofpeople_code = cursor.fetchall()
    columnDes = cursor.description
    columnNames = [columnDes[i][0] for i in range(len(columnDes))]
    df_data_plant_join_numofpeople_code = pd.DataFrame([list(i) for i in data_plant_join_numofpeople_code],
                                                       columns=columnNames)
    time_sql_end = time.time()
    print("已获取油站信息 耗时：%.3fs" % (time_sql_end - time_sql_start))

    # cursor.close()
    closeDBConnect(cursor, db)

    return df_data_plant_join_numofpeople_code

def query_plant_from_db_by_keyanliang(start_time, end_time, keyanliang):
    db = getDBConnect()
    cursor = db.cursor()

    sql_select_plant_join_numofpeople_code = "SELECT distinct a.plant , a.date, a.number_station, a.number_store, " \
                                             "b.plant_asset, b.road_class," \
                                             "b.plant_stars,b.store_class," \
                                             "b.building_area, b.business_hall," \
                                             "b.paking_area," \
                                             "b.store_area,b.plant_class_code," \
                                             "b.plant_location_class, b.plant_keyanliang_desc, b.plant_type_desc " \
                                             "from table_number_station_store  a left join table_pos_zaplant_xy_orc_508  b " \
                                             "on a.plant = b.bic_zaplant where a.date BETWEEN '" + start_time + "' and '" + end_time + "'and b.plant_keyanliang_desc = '" + keyanliang + "'  order by a.date "

    time_sql_start = time.time()
    print("正在获取油站信息......")
    cursor.execute(sql_select_plant_join_numofpeople_code)
    # 使用 fetchall() 方法获取所有数据.以元组形式返回
    data_plant_join_numofpeople_code = cursor.fetchall()
    columnDes = cursor.description
    columnNames = [columnDes[i][0] for i in range(len(columnDes))]
    df_data_plant_join_numofpeople_code = pd.DataFrame([list(i) for i in data_plant_join_numofpeople_code],
                                                       columns=columnNames)
    time_sql_end = time.time()
    print("已获取油站信息 耗时：%.3fs" % (time_sql_end - time_sql_start))

    # cursor.close()
    closeDBConnect(cursor, db)

    return df_data_plant_join_numofpeople_code


def query_plant_from_db_by_stars(start_time, end_time, stars):
    db = getDBConnect()
    cursor = db.cursor()

    sql_select_plant_join_numofpeople_code = "SELECT distinct a.plant , a.date, a.number_station, a.number_store, " \
                                             "b.plant_asset, b.road_class," \
                                             "b.plant_stars,b.store_class," \
                                             "b.building_area, b.business_hall," \
                                             "b.paking_area," \
                                             "b.store_area,b.plant_class_code," \
                                             "b.plant_location_class, b.plant_keyanliang_desc, b.plant_type_desc " \
                                             "from table_number_station_store  a left join table_pos_zaplant_xy_orc_508  b " \
                                             "on a.plant = b.bic_zaplant where a.date BETWEEN '" + start_time + "' and '" + end_time + "'and b.plant_stars_desc = '" + stars + "'  order by a.date "

    time_sql_start = time.time()
    print("正在获取油站信息......")
    cursor.execute(sql_select_plant_join_numofpeople_code)
    # 使用 fetchall() 方法获取所有数据.以元组形式返回
    data_plant_join_numofpeople_code = cursor.fetchall()
    columnDes = cursor.description
    columnNames = [columnDes[i][0] for i in range(len(columnDes))]
    df_data_plant_join_numofpeople_code = pd.DataFrame([list(i) for i in data_plant_join_numofpeople_code],
                                                       columns=columnNames)
    time_sql_end = time.time()
    print("已获取油站信息 耗时：%.3fs" % (time_sql_end - time_sql_start))

    # cursor.close()
    closeDBConnect(cursor, db)

    return df_data_plant_join_numofpeople_code


# 获取全部油站描述信息
def query_plantInfoDesc():

    sql = "select bic_zaplant as plant, plant_asset_desc as plant_asset, road_class_desc as road_class, " \
          "plant_stars_desc as plant_stars, store_class_desc as store_class, building_area, business_hall, " \
          "paking_area,store_area, plant_class_desc as plant_class_code, road_location_desc as plant_location_class, " \
          "plant_keyanliang_desc, plant_type_desc from table_pos_zaplant_xy_orc_508 "
    print("[info : databases] 正在获取油站信息")
    time_sql_start = time.time()
    db = getDBConnect()
    cursor = db.cursor()
    cursor.execute(sql)

    data_plantInfoDesc = cursor.fetchall()
    columnDes = cursor.description
    columnNames = [columnDes[i][0] for i in range(len(columnDes))]
    df_data_plantInfoDesc = pd.DataFrame([list(i) for i in data_plantInfoDesc],
                                                       columns=columnNames)
    time_sql_end = time.time()
    print("已获取油站信息 耗时：%.3fs" % (time_sql_end - time_sql_start))

    # cursor.close()
    closeDBConnect(cursor, db)
    return df_data_plantInfoDesc

# 获取分公司油站描述信息
def query_plantInfoDesc_by_company(company):

    sql = "select bic_zaplant as plant, plant_asset_desc as plant_asset, road_class_desc as road_class, " \
          "plant_stars_desc as plant_stars, store_class_desc as store_class, building_area, business_hall, " \
          "paking_area,store_area, plant_class_desc as plant_class_code, road_location_desc as plant_location_class, " \
          "plant_keyanliang_desc, plant_type_desc from table_pos_zaplant_xy_orc_508 where plant_city_desc = '" + company + "'"
    print("[info : databases] 正在获取油站信息")
    time_sql_start = time.time()
    db = getDBConnect()
    cursor = db.cursor()
    cursor.execute(sql)

    data_plantInfoDesc = cursor.fetchall()
    columnDes = cursor.description
    columnNames = [columnDes[i][0] for i in range(len(columnDes))]
    df_data_plantInfoDesc = pd.DataFrame([list(i) for i in data_plantInfoDesc],
                                                       columns=columnNames)
    time_sql_end = time.time()
    print("已获取油站信息 耗时：%.3fs" % (time_sql_end - time_sql_start))

    # cursor.close()
    closeDBConnect(cursor, db)
    return df_data_plantInfoDesc

# 获取所有油站的每日进站人数和进店人数
def get_all_numberOfPeople():

    db = getDBConnect()
    cursor = db.cursor()

    sql_select_plant_join_numofpeople_code = "SELECT distinct a.plant , a.date, a.number_station, a.number_store, " \
                                             "b.plant_asset, b.road_class," \
                                             "b.plant_stars,b.store_class," \
                                             "b.building_area, b.business_hall," \
                                             "b.paking_area," \
                                             "b.store_area,b.plant_class_code," \
                                             "b.plant_location_class, b.plant_keyanliang_desc, b.plant_type_desc " \
                                             "from table_number_station_store  a left join table_pos_zaplant_xy_orc_508  b " \
                                             "on a.plant = b.bic_zaplant where a.date BETWEEN '2019-01-01' and '2019-09-01' order by a.date"

    time_sql_start = time.time()
    print("正在获取油站信息......")
    cursor.execute(sql_select_plant_join_numofpeople_code)
    # 使用 fetchall() 方法获取所有数据.以元组形式返回
    data_plant_join_numofpeople_code = cursor.fetchall()
    columnDes = cursor.description
    columnNames = [columnDes[i][0] for i in range(len(columnDes))]
    df_data_plant_join_numofpeople_code = pd.DataFrame([list(i) for i in data_plant_join_numofpeople_code],
                                                       columns=columnNames)
    time_sql_end = time.time()
    print("已获取油站信息 耗时：%.3fs" % (time_sql_end - time_sql_start))

    # cursor.close()
    closeDBConnect(cursor, db)

    return df_data_plant_join_numofpeople_code