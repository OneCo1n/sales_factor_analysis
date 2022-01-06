import time

from db.dateBase import *


def get_bill_from_db(material, start_time, end_time):

    db = getDBConnect()
    cursor = db.cursor()


    sql_select_bill_by_material = "select  plant, `date`, quantity from table_bill_groupby_plant_calday " \
                                  " where material = '" + material + "' " \
                                  " and `date` between '" + start_time + "' and '" + end_time + "' order by `date`"
    # sql_select_bill_by_material = "select  plant, date, quantity from table_bill_groupby_plant_calmonth where material = '" + material + "' order by date"

    time_sql_start = time.time()
    print("正在获取商品销售信息......")
    cursor.execute(sql_select_bill_by_material)
    #get table
    data_material_bill = cursor.fetchall()
    #获取连接对象的描述信息
    columnDes = cursor.description
    columnNames = [columnDes[i][0] for i in range(len(columnDes))]
    df_data_material_bill = pd.DataFrame([list(i) for i in data_material_bill], columns=columnNames)
    time_sql_end = time.time()

    print("已获取商品销售信息 耗时：%.3fs" % (time_sql_end - time_sql_start))
    # 关闭数据库连接
    closeDBConnect(cursor,db)

    return df_data_material_bill

def get_bill_by_category_from_db(category, start_time, end_time):
    db = getDBConnect()
    cursor = db.cursor()

    sql_select_bill_by_material = "select material, plant, `date`, quantity from table_bill_groupby_plant_calday " \
                                  " where material in (" \
                                  " select material from table_material_category " \
                                  " WHERE big_category_name = '" + category + "') " \
                                  " and `date` between '" + start_time + "' and '" + end_time + "' order by `date`"
    # sql_select_bill_by_material = "select  plant, date, quantity from table_bill_groupby_plant_calmonth where material = '" + material + "' order by date"

    time_sql_start = time.time()
    print("正在获取商品销售信息......")
    cursor.execute(sql_select_bill_by_material)
    # get table
    data_material_bill = cursor.fetchall()
    # 获取连接对象的描述信息
    columnDes = cursor.description
    columnNames = [columnDes[i][0] for i in range(len(columnDes))]
    df_data_material_bill = pd.DataFrame([list(i) for i in data_material_bill], columns=columnNames)
    time_sql_end = time.time()

    print("已获取商品销售信息 耗时：%.3fs" % (time_sql_end - time_sql_start))
    # 关闭数据库连接
    closeDBConnect(cursor, db)

    return df_data_material_bill


def get_bill_from_db_by_material_and_company(material, start_time, end_time, company):

    db = getDBConnect()
    cursor = db.cursor()

    sql_select_bill_by_material = "select  plant, `date`, quantity from table_bill_groupby_plant_calday " \
                                  "where material = '" + material + "' and `date` between '" + start_time + "' and '" + end_time + "' " \
                                  "and plant in (SELECT plant from table_pos_zaplant_xy_orc_508 " \
                                                                    "where plant_city_desc = '" + company + "')" \
                                  "order by `date`"
    time_sql_start = time.time()
    print("正在获取商品销售信息......")
    cursor.execute(sql_select_bill_by_material)
    #get table
    data_material_bill = cursor.fetchall()
    #获取连接对象的描述信息
    columnDes = cursor.description
    columnNames = [columnDes[i][0] for i in range(len(columnDes))]
    df_data_material_bill = pd.DataFrame([list(i) for i in data_material_bill], columns=columnNames)
    time_sql_end = time.time()

    print("已获取商品销售信息 耗时：%.3fs" % (time_sql_end - time_sql_start))
    # 关闭数据库连接
    closeDBConnect(cursor,db)

    return df_data_material_bill

def get_bill_from_db_by_category_and_company(category, start_time, end_time, company):

    db = getDBConnect()
    cursor = db.cursor()

    sql_select_bill_by_material = "select material, plant, `date`, quantity from table_bill_groupby_plant_calday " \
                                  " where material in (" \
                                  " select material from table_material_category " \
                                  " WHERE big_category_name = '" + category + "') " \
                                  " and `date` between '" + start_time + "' and '" + end_time + "' " \
                                  " and plant in (SELECT plant from table_pos_zaplant_xy_orc_508 " \
                                                                    "where plant_city_desc = '" + company + "')" \
                                  "order by `date`"
    time_sql_start = time.time()
    print("正在获取商品销售信息......")
    cursor.execute(sql_select_bill_by_material)
    #get table
    data_material_bill = cursor.fetchall()
    #获取连接对象的描述信息
    columnDes = cursor.description
    columnNames = [columnDes[i][0] for i in range(len(columnDes))]
    df_data_material_bill = pd.DataFrame([list(i) for i in data_material_bill], columns=columnNames)
    time_sql_end = time.time()

    print("已获取商品销售信息 耗时：%.3fs" % (time_sql_end - time_sql_start))
    # 关闭数据库连接
    closeDBConnect(cursor,db)

    return df_data_material_bill


def get_bill_from_db_by_category_and_keyanliang(category, start_time, end_time, keyanliang):

    db = getDBConnect()
    cursor = db.cursor()

    sql_select_bill_by_material = "select material, plant, `date`, quantity from table_bill_groupby_plant_calday " \
                                  " where material in (" \
                                  " select material from table_material_category " \
                                  " WHERE big_category_name = '" + category + "') " \
                                  " and `date` between '" + start_time + "' and '" + end_time + "' " \
                                  " and plant in (SELECT plant from table_pos_zaplant_xy_orc_508 " \
                                                                    " where plant_keyanliang_desc = '" + keyanliang + "')" \
                                  "order by `date`"
    time_sql_start = time.time()
    print("正在获取商品销售信息......")
    cursor.execute(sql_select_bill_by_material)
    #get table
    data_material_bill = cursor.fetchall()
    #获取连接对象的描述信息
    columnDes = cursor.description
    columnNames = [columnDes[i][0] for i in range(len(columnDes))]
    df_data_material_bill = pd.DataFrame([list(i) for i in data_material_bill], columns=columnNames)
    time_sql_end = time.time()

    print("已获取商品销售信息 耗时：%.3fs" % (time_sql_end - time_sql_start))
    # 关闭数据库连接
    closeDBConnect(cursor,db)

    return df_data_material_bill


def get_bill_from_db_by_category_and_stars(category, start_time, end_time, stars):

    db = getDBConnect()
    cursor = db.cursor()

    sql_select_bill_by_material = "select material, plant, `date`, quantity from table_bill_groupby_plant_calday " \
                                  " where material in (" \
                                  " select material from table_material_category " \
                                  " WHERE big_category_name = '" + category + "') " \
                                  " and `date` between '" + start_time + "' and '" + end_time + "' " \
                                  " and plant in (SELECT plant from table_pos_zaplant_xy_orc_508 " \
                                                                    " where plant_stars_desc = '" + stars + "')" \
                                  "order by `date`"
    time_sql_start = time.time()
    print("正在获取商品销售信息......")
    cursor.execute(sql_select_bill_by_material)
    #get table
    data_material_bill = cursor.fetchall()
    #获取连接对象的描述信息
    columnDes = cursor.description
    columnNames = [columnDes[i][0] for i in range(len(columnDes))]
    df_data_material_bill = pd.DataFrame([list(i) for i in data_material_bill], columns=columnNames)
    time_sql_end = time.time()

    print("已获取商品销售信息 耗时：%.3fs" % (time_sql_end - time_sql_start))
    # 关闭数据库连接
    closeDBConnect(cursor,db)

    return df_data_material_bill


def get_bill_from_db_by_material_and_keyanliang(material, start_time, end_time, keyanliang):

    db = getDBConnect()
    cursor = db.cursor()

    sql_select_bill_by_material = "select  plant, `date`, quantity from table_bill_groupby_plant_calday " \
                                  "where material = '" + material + "' and `date` between '" + start_time + "' and '" + end_time + "' " \
                                  "and plant in (SELECT plant from table_pos_zaplant_xy_orc_508 " \
                                                                    " where plant_keyanliang_desc= '" + keyanliang + "')" \
                                  "order by `date`"
    time_sql_start = time.time()
    print("正在获取商品销售信息......")
    cursor.execute(sql_select_bill_by_material)
    #get table
    data_material_bill = cursor.fetchall()
    #获取连接对象的描述信息
    columnDes = cursor.description
    columnNames = [columnDes[i][0] for i in range(len(columnDes))]
    df_data_material_bill = pd.DataFrame([list(i) for i in data_material_bill], columns=columnNames)
    time_sql_end = time.time()

    print("已获取商品销售信息 耗时：%.3fs" % (time_sql_end - time_sql_start))
    # 关闭数据库连接
    closeDBConnect(cursor,db)

    return df_data_material_bill

def get_bill_from_db_by_material_and_stars(material, start_time, end_time, stars):

    db = getDBConnect()
    cursor = db.cursor()

    sql_select_bill_by_material = "select  plant, `date`, quantity from table_bill_groupby_plant_calday " \
                                  "where material = '" + material + "' and `date` between '" + start_time + "' and '" + end_time + "' " \
                                  "and plant in (SELECT plant from table_pos_zaplant_xy_orc_508 " \
                                                                    " where plant_stars_desc= '" + stars + "')" \
                                  "order by `date`"
    time_sql_start = time.time()
    print("正在获取商品销售信息......")
    cursor.execute(sql_select_bill_by_material)
    #get table
    data_material_bill = cursor.fetchall()
    #获取连接对象的描述信息
    columnDes = cursor.description
    columnNames = [columnDes[i][0] for i in range(len(columnDes))]
    df_data_material_bill = pd.DataFrame([list(i) for i in data_material_bill], columns=columnNames)
    time_sql_end = time.time()

    print("已获取商品销售信息 耗时：%.3fs" % (time_sql_end - time_sql_start))
    # 关闭数据库连接
    closeDBConnect(cursor,db)

    return df_data_material_bill

def queryDailySalesByMaterialAndPlant(material, plant):
    db = getDBConnect()
    cursor = db.cursor()

    sql_select_bill_by_material = "select  `plant`, `date`, quantity from table_bill_groupby_plant_calday where material = '" + material + "' and plant = '" + plant + "' and `date` between '2019-01-01' and '2019-09-01' order by `date`"
    # sql_select_bill_by_material = "select  plant, date, quantity from table_bill_groupby_plant_calmonth where material = '" + material + "' order by date"

    time_sql_start = time.time()
    print("正在获取商品销售信息......")
    cursor.execute(sql_select_bill_by_material)
    # get table
    data_material_bill = cursor.fetchall()
    # 获取连接对象的描述信息
    columnDes = cursor.description
    columnNames = [columnDes[i][0] for i in range(len(columnDes))]
    df_data_material_bill = pd.DataFrame([list(i) for i in data_material_bill], columns=columnNames)
    time_sql_end = time.time()

    print("已获取商品销售信息 耗时：%.3fs" % (time_sql_end - time_sql_start))
    # 关闭数据库连接
    closeDBConnect(cursor, db)

    return df_data_material_bill
