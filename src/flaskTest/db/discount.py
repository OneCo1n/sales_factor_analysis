
import time

from db.dateBase import *

def get_discount_from_db_by_material(material, start_time, end_time):

    db = getDBConnect()
    cursor = db.cursor()


    sql_select_discount_by_material = "SELECT  a.plant, a.calday, a.discount_rate " \
                                      "from table_discount_rate a " \
                                      "WHERE a.material = '" + material + "' and a.calday between '"+ start_time + "' and '" + end_time + "'" \
                                                                        "order by a.calday"
    time_sql_start = time.time()
    print("正在获取商品折扣......")
    cursor.execute(sql_select_discount_by_material)
    data_material_discount = cursor.fetchall()
    columnDes = cursor.description
    columnNames = [columnDes[i][0] for i in range(len(columnDes))]
    df_data_material_discount = pd.DataFrame([list(i) for i in data_material_discount], columns=columnNames)
    time_sql_end = time.time()
    print("已获取商品折扣 耗时：%.3fs" % (time_sql_end - time_sql_start))

    # 关闭数据库连接
    closeDBConnect(cursor,db)

    return df_data_material_discount

