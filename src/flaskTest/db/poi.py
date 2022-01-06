import time

from db.dateBase import *

def get_poi_from_db():

    db = getDBConnect()
    cursor = db.cursor()

    sql_select_plant_poi = "select distinct * from table_plant_poi"

    time_sql_start = time.time()
    print("正在获取油站周围3000m之内的poi信息......")
    cursor.execute(sql_select_plant_poi)
    # 使用 fetchall() 方法获取所有数据.以元组形式返回
    data_plant_poi = cursor.fetchall()
    columnDes = cursor.description
    columnNames = [columnDes[i][0] for i in range(len(columnDes))]
    df_plant_poi = pd.DataFrame([list(i) for i in data_plant_poi],
                                                   columns=columnNames)
    time_sql_end = time.time()
    print("已获取poi信息 耗时：%.3fs" % (time_sql_end - time_sql_start))
    #cursor.close()
    closeDBConnect(cursor,db)

    return df_plant_poi
