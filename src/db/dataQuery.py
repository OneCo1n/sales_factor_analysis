from db.dateBase import *
import time

def getMaterialsId():

    sql = "SELECT distinct material from bill_zsd where calday BETWEEN '2016-01-01' and '2021-06-01'"
    time_start = time.time()
    print("正在获取商品Id列表......")
    materialsId = exe_sql(sql)
    print("已获取商品Id列表 耗时：%.3fs" % (time.time() - time_start))

    return materialsId