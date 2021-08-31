import pymysql
import pandas as pd
import time
import numpy as np

from data_preprocess.dtype_exchange import date_conversion_toStr
from data_preprocess.del_str import del_str_m2

from data_preprocess.missing_value_processing import *
from db.dateBase import *


def get_df_from_db(material):
#pd.read_sql(sql,connection)
# 使用 cursor() 方法创建一个游标对象 cursor
    db = getDBConnect()
    cursor = db.cursor()
# 使用 execute()  方法执行 SQL 查询
    print("start ")

    #material = "0000000000" + material
    sql_select_bill_by_material = "select  plant, date, quantity from table_bill_groupby_plant_calday where material = '" + material + "'"

    sql_select_discount_by_material = "SELECT  a.plant, a.calday, a.discount_rate " \
                                      "from table_discount_rate a " \
                                      "WHERE a.material = '" + material + "' and a.calday between '2016-01-01' and '2021-06-01'" \
                                                                        "order by a.calday"
    sql_select_promotion_by_material = "select plant, promotion_type, discount, promotion_quantity, promotion_amount, retail_price, calday " \
                                       "from table_promotion " \
                                       "WHERE material = '" + material + "' and calday BETWEEN '2016-01-01' and '2021-06-01'"


    sql_select_plant_join_numofpeople = "SELECT a.plant , a.date, a.number_station, a.number_store, " \
                                        "b.plant_asset, b.road_class," \
                                        "b.plant_stars,b.store_class," \
                                        "b.street,b.building_area, b.business_hall," \
                                        "b.paking_area,b.surrounding_building," \
                                        "b.store_area,b.plant_class_code," \
                                        "b.plant_location_class " \
                                        "from table_number_station_store  a left join table_pos_zaplant_xy_orc_508  b " \
                                        "on a.plant = b.bic_zaplant where a.date BETWEEN '2016-01-01' and '2021-06-01' order by a.date"

    sql_select_plant_join_numofpeople_code = "SELECT a.plant , a.date, a.number_station, a.number_store, " \
                                        "b.plant_asset, b.road_class," \
                                        "b.plant_stars,b.store_class," \
                                        "b.building_area, b.business_hall," \
                                        "b.paking_area," \
                                        "b.store_area,b.plant_class_code," \
                                        "b.plant_location_class, b.plant_keyanliang_desc, b.plant_type_desc " \
                                        "from table_number_station_store  a left join table_pos_zaplant_xy_orc_508  b " \
                                        "on a.plant = b.bic_zaplant where a.date BETWEEN '2016-01-01' and '2021-06-01' order by a.date"

    #cursor.execute("select * from table_70251989_sales_plant")
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
    # outputpath='F:\论文\data\\df_data_material_bill.csv'
    # df_data_material_bill.to_csv(outputpath,index=False,header=True)
    print("已获取商品销售信息 耗时：%.3fs" % (time_sql_end - time_sql_start))

    time_sql_start = time.time()
    print("正在获取商品折扣......")
    cursor.execute(sql_select_discount_by_material)
    data_material_discount = cursor.fetchall()
    columnDes = cursor.description
    columnNames = [columnDes[i][0] for i in range(len(columnDes))]
    df_data_material_discount = pd.DataFrame([list(i) for i in data_material_discount], columns=columnNames)
    time_sql_end = time.time()
    # outputpath='F:\论文\data\\df_data_material_discount.csv'
    # df_data_material_discount.to_csv(outputpath,index=False,header=True)
    print("已获取商品折扣 耗时：%.3fs" % (time_sql_end - time_sql_start))

    time_sql_start = time.time()
    print("正在获取商品促销信息......")
    cursor.execute(sql_select_promotion_by_material)
    data_material_promotion = cursor.fetchall()
    columnDes = cursor.description
    columnNames = [columnDes[i][0] for i in range(len(columnDes))]
    df_data_material_promotion = pd.DataFrame([list(i) for i in data_material_promotion], columns=columnNames)
    time_sql_end = time.time()
    # outputpath='F:\论文\data\\df_data_material_promotion.csv'
    # df_data_material_promotion.to_csv(outputpath,index=False,header=True)
    print("已获取商品促销信息 耗时：%.3fs" % (time_sql_end - time_sql_start))

    # time_sql_start = time.time()
    # print("正在获取油站信息......")
    # cursor.execute(sql_select_plant_join_numofpeople)
    # # 使用 fetchall() 方法获取所有数据.以元组形式返回
    # data_plant_join_numofpeople = cursor.fetchall()
    # columnDes = cursor.description
    # columnNames = [columnDes[i][0] for i in range(len(columnDes))]
    # df_data_plant_join_numofpeople = pd.DataFrame([list(i) for i in data_plant_join_numofpeople], columns=columnNames)
    # time_sql_end = time.time()
    # outputpath='F:\论文\data\\df_data_plant_join_numofpeople.csv'
    # df_data_plant_join_numofpeople.to_csv(outputpath,index=False,header=True)
    # print("已获取油站信息 耗时：%.3fs" % (time_sql_end - time_sql_start))

    time_sql_start = time.time()
    print("正在获取油站信息......")
    cursor.execute(sql_select_plant_join_numofpeople_code)
    # 使用 fetchall() 方法获取所有数据.以元组形式返回
    data_plant_join_numofpeople_code = cursor.fetchall()
    columnDes = cursor.description
    columnNames = [columnDes[i][0] for i in range(len(columnDes))]
    df_data_plant_join_numofpeople_code = pd.DataFrame([list(i) for i in data_plant_join_numofpeople_code], columns=columnNames)
    time_sql_end = time.time()
    # outputpath='F:\论文\data\\df_data_plant_join_numofpeople_code.csv'
    # df_data_plant_join_numofpeople_code.to_csv(outputpath,index=False,header=True)
    print("已获取油站信息 耗时：%.3fs" % (time_sql_end - time_sql_start))

    #print("execute time: %.3f" % ((time_sql_end - time_sql_start) ) )

    #print(data)
# 关闭数据库连接
    cursor.close()
    closeDBConnect(db)
    #print(df)

    # df_data_material_bill.columns=('plant', 'date', 'quantity')
    # df_data_material_discount.columns=('plant', 'calday', 'discount_rate')
    # df_data_material_promotion.columns=('plant', 'promotion_type', 'promotion_desc', 'discount', 'promotion_quantity', 'promotion_amount', 'retail_price', 'calday' )
    # df_data_plant_join_numofpeople.columns=('plant', 'date', 'number_station', 'number_store', 'plant_asset', 'road_class', 'plant_stars', 'store_class',
    #                                         'building_area', 'business_hall', 'paking_area',  'store_area', 'plant_class_code', 'plant_location_class,
    #                                         plant_location_class, plant_keyanliang_desc, plant_type_desc')





#
#                                           更改列名
#
    df_data_material_promotion.rename(columns = {'calday':'date'}, inplace=True)
    df_data_material_discount.rename(columns = {'calday':'date'}, inplace=True)

################################################################################################################################
#                                   查看缺失情况
#
################################################################################################################################
    # print("-------------------------缺失值检测-----------------------------")
    # missing_value_check(df_data_material_bill)
    # missing_value_check(df_data_material_discount)
    # missing_value_check(df_data_material_promotion)
    # missing_value_check(df_data_plant_join_numofpeople_code)

    #删除面积字段中多余字符串
    del_str_m2(df_data_plant_join_numofpeople_code, 'building_area')
    del_str_m2(df_data_plant_join_numofpeople_code, 'business_hall')
    del_str_m2(df_data_plant_join_numofpeople_code, 'paking_area')
    del_str_m2(df_data_plant_join_numofpeople_code, 'store_area')







#################################################################################################################################
#                                 数据类型转换
#                         更改dataframe字段类型，否则数据合并会出现大量NAN
##################################################################################################################################


    print("数据预处理：数据转换 start......")
    time_data_conversion_start = time.time()

    #全部变为str
    # df_data_material_bill=date_conversion_toStr(df_data_material_bill)
    # df_data_material_discount=date_conversion_toStr(df_data_material_discount)
    # df_data_material_promotion=date_conversion_toStr(df_data_material_promotion)
    # df_data_plant_join_numofpeople_code=date_conversion_toStr(df_data_plant_join_numofpeople_code)


    df_data_material_bill['plant']=df_data_material_bill['plant'].astype('str')
    df_data_material_bill['date']=df_data_material_bill['date'].astype('datetime64')
    df_data_material_bill['quantity']=df_data_material_bill['quantity'].astype('int')

    #df_data_material_discount.columns=('plant', 'calday', 'discount_rate')
    df_data_material_discount['plant']=df_data_material_discount['plant'].astype('str')
    df_data_material_discount['date'] = df_data_material_discount['date'].astype('datetime64')
    df_data_material_discount['discount_rate'] = df_data_material_discount['discount_rate'].astype('float64')

    # df_data_material_promotion.columns=('plant', 'promotion_type', 'discount', 'promotion_quantity', 'promotion_amount', 'retail_price', 'calday' )
    df_data_material_promotion['plant'] = df_data_material_promotion['plant'].astype('str')
    df_data_material_promotion['promotion_type'] = df_data_material_promotion['promotion_type'].astype('str')
    df_data_material_promotion['discount'] = df_data_material_promotion['discount'].astype('float64')
    df_data_material_promotion['promotion_quantity'] = df_data_material_promotion['promotion_quantity'].astype('float64')
    df_data_material_promotion['promotion_quantity'] = df_data_material_promotion['promotion_quantity'].astype('int')
    df_data_material_promotion['promotion_amount'] = df_data_material_promotion['promotion_amount'].astype('float64')
    df_data_material_promotion['retail_price'] = df_data_material_promotion['retail_price'].astype('float64')
    df_data_material_promotion['date'] = df_data_material_promotion['date'].astype('datetime64')

    #############################################
    ##处理含有负数的记录
    #df_data_material_promotion = df_data_material_promotion.abs()
    df_data_material_promotion['discount']=df_data_material_promotion['discount'].abs()
    df_data_material_promotion['promotion_amount']=df_data_material_promotion['promotion_amount'].abs()
    df_data_material_promotion['retail_price']=df_data_material_promotion['retail_price'].abs()
    df_data_material_promotion['promotion_amount'], df_data_material_promotion['retail_price']= np.where(df_data_material_promotion['promotion_amount'] > df_data_material_promotion['retail_price'], [df_data_material_promotion['retail_price'], df_data_material_promotion['promotion_amount']], [df_data_material_promotion['promotion_amount'], df_data_material_promotion['retail_price']])
    df_data_material_promotion['discount_rate2'] = (df_data_material_promotion['promotion_amount'] + 0.00001) / (df_data_material_promotion['retail_price'] + 0.0001)




    # df_data_plant_join_numofpeople.columns=('plant', 'date', 'number_station', 'number_store', 'plant_asset', 'road_class', 'plant_stars', 'store_class',
    #                                         'building_area', 'business_hall', 'paking_area',  'store_area', 'plant_class_code', 'plant_location_class,
    #                                         plant_location_class, plant_keyanliang_desc, plant_type_desc')
    for i in df_data_plant_join_numofpeople_code.columns[1:]:
        df_data_plant_join_numofpeople_code[i] = df_data_plant_join_numofpeople_code[i].astype('str')


    df_data_plant_join_numofpeople_code['date'] = df_data_plant_join_numofpeople_code['date'].astype('datetime64')
    df_data_plant_join_numofpeople_code['number_station'] = df_data_plant_join_numofpeople_code['number_station'].astype('int')
    df_data_plant_join_numofpeople_code['number_store'] = df_data_plant_join_numofpeople_code['number_store'].astype('int')
    df_data_plant_join_numofpeople_code['building_area'] = df_data_plant_join_numofpeople_code['building_area'].astype('float64')
    df_data_plant_join_numofpeople_code['business_hall'] = df_data_plant_join_numofpeople_code['business_hall'].astype('float64')
    df_data_plant_join_numofpeople_code['paking_area'] = df_data_plant_join_numofpeople_code['paking_area'].astype('float64')
    df_data_plant_join_numofpeople_code['store_area'] = df_data_plant_join_numofpeople_code['store_area'].astype('float64')



    time_data_conversion_end = time.time()
    print("数据转换 - 1 完成 耗时：%.3fs" % (time_data_conversion_end - time_data_conversion_start))




#######################################################################################################################
#                                               异常值处理
#
#####################################################################################################################

    df_data_material_promotion[(df_data_material_promotion['discount_rate2'] < 0)&(df_data_material_promotion['discount_rate2'] > 1)]
    #nan_value_check(df_data_material_bill)
    #nan_value_check(df_data_material_discount)
    #nan_value_check(df_data_material_promotion)
    #nan_value_check(df_data_plant_join_numofpeople_code)


###################################################################################################################################
#                                           数据合并
#
#################################################################################################################################
    time_data_merge_start = time.time()
    print("数据预处理：数据合并start......" )
    # df_material_sales = pd.DataFrame(columns=( 'plant', 'date', 'quantity', 'discount_rate'))
    # for i in df_material_sales[1:]:
    #   df_material_sales[i] = df_material_sales[i].astype('str')
    #
    # #df_material_sales['plant']=df_material_sales['plant'].astype('str')
    # df_material_sales['discount_rate']=df_material_sales['discount_rate'].astype('float64')

    #df_data_material_discount.rename(columns = {'calday':'date'}, inplace=True)
    #数据合并
    #columns=('material', 'plant', 'date', 'quantity', 'discount_rate')

    #df_material_sales = pd.merge(df_data_material_bill, df_data_material_discount, left_on = ['plant','date'], right_on = ['plant', 'date'], how = 'left' )
    df_material_sales = pd.merge(df_data_material_bill, df_data_material_promotion, left_on = ['plant','date'], right_on = ['plant', 'date'], how = 'left' )

    #print(df_data_material_bill)
    #print(df_data_material_discount)
    #df_material_sales = pd.merge(df_data_material_bill, df_data_material_discount,on = ['material' ,'plant' ,'date'] ,how = 'left' )
    #print(df_material_sales.dtypes)


    #更改dataframe字段类型，否则数据合并会出现大量NAN
    # df_data_plant_join_numofpeople['plant'] = df_data_plant_join_numofpeople['plant'].astype('str')
    # df_data_plant_join_numofpeople['date'] = df_data_plant_join_numofpeople['date'].astype('str')
    #数据合并
    df_data_material_discount_plant = pd.merge(df_material_sales, df_data_plant_join_numofpeople_code, left_on=['plant' ,'date'], right_on = ['plant', 'date'], how = 'left')
    #df = pd.merge(df_data_material_discount_plant, df_data_material_promotion, left_on=['plant', 'date'], right_on=['plant', 'date'], how='left')
    time_data_merge_end = time.time()

    print("数据合并完成 耗时：%.3fs" % (time_data_merge_end - time_data_merge_start))


    ##合并后的数据进行缺失值检测
    # missing_value_check(df_data_material_discount_plant)
    #nan_value_check(df_data_material_discount_plant)




######################################################################################################################
#                                               缺失值处理
#
######################################################################################################################
    print("数据预处理：缺失值处理 start......")

    time_miss_value_start = time.time()
    ####对合并后的数据进行缺失值处理

    #将none变为nan方便后面对空值的处理
    #df_data_material_discount_plant = df_data_material_discount_plant.fillna(value=np.nan)
    df_data_material_discount_plant.replace('None', np.nan, inplace= True)
    col_set = ['number_station', 'number_store']
    ##删除number_station、number_store为空的行
    del_missing_value_row(df_data_material_discount_plant, col_set)

    #df_data_material_promotion缺失值处理
    missing_value_fill_str_NP(df_data_material_discount_plant, 'promotion_type')
    missing_value_fill_num_0(df_data_material_discount_plant, 'discount')
    missing_value_fill_num_0(df_data_material_discount_plant, 'promotion_quantity')
    missing_value_fill_num_0(df_data_material_discount_plant, 'promotion_amount')
    missing_value_fill_num_0(df_data_material_discount_plant, 'retail_price')
    missing_value_fill_num_1(df_data_material_discount_plant, 'discount_rate2')


    #缺失值处理之后的情况
    print('-----------------缺失值处理之后------------------')
    # missing_value_check(df_data_material_bill)
    # missing_value_check(df_data_material_discount)
    # missing_value_check(df_data_material_promotion)
    # missing_value_check(df_data_plant_join_numofpeople_code)
    time_miss_value_end = time.time()
    print("已处理缺失值 耗时：%.3fs" % (time_miss_value_end - time_miss_value_start))

    #缺失值检测
    #missing_value_check(df_data_material_discount_plant)





#
# ##################################################################################################################################
# #                                  数据类型转换 - 2
# #                          数据类型转换，之后好进行编码
# ###################################################################################################################################
#
#     print("数据预处理：数据转换 - 2 start......")
#     time_data_conversion_start = time.time()
#
#     #df_data_material_promotion.rename(columns = {'calday':'date'}, inplace=True)
#     #print(df_data_material_bill.dtypes)
#     #print(df_data_material_discount.dtypes)
#     #更改dataframe列类型，避免联接之后字段出现大量NAN
#
#     df_data_material_bill['plant']=df_data_material_bill['plant'].astype('str')
#     df_data_material_bill['date']=df_data_material_bill['date'].astype('datetime64')
#     df_data_material_bill['quantity']=df_data_material_bill['quantity'].astype('int')
#
#
#     # df_data_material_discount.columns=('plant', 'calday', 'discount_rate')
#     df_data_material_discount.rename(columns = {'calday':'date'}, inplace=True)
#
#     df_data_material_discount['plant']=df_data_material_discount['plant'].astype('str')
#     df_data_material_discount['date'] = df_data_material_discount['date'].astype('datetime64')
#     df_data_material_discount['discount_rate'] = df_data_material_discount['discount_rate'].astype('float64')
#
#     # df_data_material_promotion.columns=('plant', 'promotion_type', 'discount', 'promotion_quantity', 'promotion_amount', 'retail_price', 'calday' )
#     df_data_material_promotion.rename(columns = {'calday':'date'}, inplace=True)
#     df_data_material_promotion['plant'] = df_data_material_promotion['plant'].astype('str')
#     df_data_material_promotion['promotion_type'] = df_data_material_promotion['promotion_type'].astype('str')
#     df_data_material_promotion['discount'] = df_data_material_promotion['discount'].astype('float64')
#     df_data_material_promotion['promotion_quantity'] = df_data_material_promotion['promotion_quantity'].astype('int')
#     df_data_material_promotion['promotion_amount'] = df_data_material_promotion['promotion_amount'].astype('float64')
#     df_data_material_promotion['retail_price'] = df_data_material_promotion['retail_price'].astype('float64')
#     df_data_material_promotion['date'] = df_data_material_promotion['date'].astype('datetime64')
#
#     # df_data_plant_join_numofpeople.columns=('plant', 'date', 'number_station', 'number_store', 'plant_asset', 'road_class', 'plant_stars', 'store_class',
#     #                                         'building_area', 'business_hall', 'paking_area',  'store_area', 'plant_class_code', 'plant_location_class,
#     #                                         plant_location_class, plant_keyanliang_desc, plant_type_desc')
#     for i in df_data_plant_join_numofpeople_code.columns[1:]:
#         df_data_plant_join_numofpeople_code[i] = df_data_plant_join_numofpeople_code[i].astype('str')
#     #删除面积字段中多余字符串
#     del_str_m2(df_data_plant_join_numofpeople_code)
#     df_data_plant_join_numofpeople_code['date'] = df_data_plant_join_numofpeople_code['date'].astype('datetime64')
#     df_data_plant_join_numofpeople_code['number_station'] = df_data_plant_join_numofpeople_code['number_station'].astype('int')
#     df_data_plant_join_numofpeople_code['number_store'] = df_data_plant_join_numofpeople_code['number_store'].astype('int')
#     df_data_plant_join_numofpeople_code['building_area'] = df_data_plant_join_numofpeople_code['building_area'].astype('float64')
#     df_data_plant_join_numofpeople_code['business_hall'] = df_data_plant_join_numofpeople_code['business_hall'].astype('float64')
#     df_data_plant_join_numofpeople_code['paking_area'] = df_data_plant_join_numofpeople_code['paking_area'].astype('float64')
#     df_data_plant_join_numofpeople_code['store_area'] = df_data_plant_join_numofpeople_code['store_area'].astype('float64')
#
#     time_data_conversion_end = time.time()
#     print("数据转换完成 耗时：%.3fs" % (time_data_conversion_end - time_data_conversion_start))
#     #print(df_data_material_bill.dtypes)
#     #print(df_data_material_discount.dtypes)
#     #print(df_data_material_promotion)

    # for i in df_data_material_discount_plant.columns:
    #     print(df_data_material_discount_plant[i].unique())


    return df_data_material_discount_plant
    #return df