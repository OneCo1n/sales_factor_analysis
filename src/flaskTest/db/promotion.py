import time

from db.dateBase import *

def get_promotion_from_db_by_material(material, start_time, end_time):

    db = getDBConnect()
    cursor = db.cursor()

    sql_select_promotion_by_material = "select plant, promotion_type, discount, promotion_quantity, promotion_amount, retail_price, calday " \
                                       " from table_promotion " \
                                       " WHERE material = '" + material + "' and calday BETWEEN '" + start_time + "' and '" + end_time + "'"
    time_sql_start = time.time()
    print("正在获取商品促销信息......")
    cursor.execute(sql_select_promotion_by_material)
    data_material_promotion = cursor.fetchall()
    columnDes = cursor.description
    columnNames = [columnDes[i][0] for i in range(len(columnDes))]
    df_data_material_promotion = pd.DataFrame([list(i) for i in data_material_promotion], columns=columnNames)
    time_sql_end = time.time()
    print("已获取商品促销信息 耗时：%.3fs" % (time_sql_end - time_sql_start))
    #cursor.close()
    closeDBConnect(cursor,db)

    return df_data_material_promotion

def get_promotion_from_db_by_all_material_and_plant_class(start_time, end_time, plant_class):
    db = getDBConnect()
    cursor = db.cursor()

    sql_select_promotion_by_material = "select material, plant, promotion_type, discount, promotion_quantity, promotion_amount, retail_price, calday " \
                                       " from table_promotion " \
                                       " WHERE calday BETWEEN '" + start_time + "' and '" + end_time + "' " \
                                       " and plant in ( select DISTINCT bic_zaplant " \
                                                      " from table_pos_zaplant_xy_orc_508 " \
                                                      " where plant_class_code = '" + plant_class + "')"
    time_sql_start = time.time()
    print("正在获取商品促销信息......")
    cursor.execute(sql_select_promotion_by_material)
    data_material_promotion = cursor.fetchall()
    columnDes = cursor.description
    columnNames = [columnDes[i][0] for i in range(len(columnDes))]
    df_data_material_promotion = pd.DataFrame([list(i) for i in data_material_promotion], columns=columnNames)
    time_sql_end = time.time()
    print("已获取商品促销信息 耗时：%.3fs" % (time_sql_end - time_sql_start))
    #cursor.close()
    closeDBConnect(cursor,db)

    return df_data_material_promotion

def get_promotion_from_db_by_category(category, start_time, end_time):

    db = getDBConnect()
    cursor = db.cursor()

    sql_select_promotion_by_material = "select material, plant, promotion_type, discount, promotion_quantity, promotion_amount, retail_price, calday " \
                                       " from table_promotion " \
                                       " WHERE material in (" \
                                       " select material from table_material_category " \
                                       " WHERE big_category_name = '" + category + "') " \
                                       " and calday BETWEEN '" + start_time + "' and '" + end_time + "'"
    time_sql_start = time.time()
    print("正在获取商品促销信息......")
    cursor.execute(sql_select_promotion_by_material)
    data_material_promotion = cursor.fetchall()
    columnDes = cursor.description
    columnNames = [columnDes[i][0] for i in range(len(columnDes))]
    df_data_material_promotion = pd.DataFrame([list(i) for i in data_material_promotion], columns=columnNames)
    time_sql_end = time.time()
    print("已获取商品促销信息 耗时：%.3fs" % (time_sql_end - time_sql_start))
    #cursor.close()
    closeDBConnect(cursor,db)

    return df_data_material_promotion


def get_promotion_desc_from_db_by_material(material, start_time, end_time):
    db = getDBConnect()
    cursor = db.cursor()

    sql_select_promotion_by_material = "select plant, promotion_type,promotion_desc,  discount, promotion_quantity, promotion_amount, retail_price, calday as `date` " \
                                       " from table_promotion " \
                                       " WHERE material = '" + material + "' and calday BETWEEN '" + start_time + "' and '" + end_time + "'"
    time_sql_start = time.time()
    print("正在获取商品促销信息......")
    cursor.execute(sql_select_promotion_by_material)
    data_material_promotion = cursor.fetchall()
    columnDes = cursor.description
    columnNames = [columnDes[i][0] for i in range(len(columnDes))]
    df_data_material_promotion = pd.DataFrame([list(i) for i in data_material_promotion], columns=columnNames)
    time_sql_end = time.time()
    print("已获取商品促销信息 耗时：%.3fs" % (time_sql_end - time_sql_start))
    # cursor.close()
    closeDBConnect(cursor, db)

    return df_data_material_promotion

def get_promotion_desc_from_db_by_category(category, start_time, end_time):
    db = getDBConnect()
    cursor = db.cursor()

    sql_select_promotion_by_material = "select plant, promotion_type,promotion_desc,  discount, promotion_quantity, promotion_amount, retail_price, calday as `date` " \
                                       " from table_promotion " \
                                       " WHERE material  in (" \
                                       " select material from table_material_category " \
                                       " WHERE big_category_name = '" + category + "') " \
                                       " and calday BETWEEN '" + start_time + "' and '" + end_time + "'"
    time_sql_start = time.time()
    print("正在获取商品促销信息......")
    cursor.execute(sql_select_promotion_by_material)
    data_material_promotion = cursor.fetchall()
    columnDes = cursor.description
    columnNames = [columnDes[i][0] for i in range(len(columnDes))]
    df_data_material_promotion = pd.DataFrame([list(i) for i in data_material_promotion], columns=columnNames)
    time_sql_end = time.time()
    print("已获取商品促销信息 耗时：%.3fs" % (time_sql_end - time_sql_start))
    # cursor.close()
    closeDBConnect(cursor, db)

    return df_data_material_promotion
