import pymysql
import pandas as pd


def getDBConnect():
    host = "10.112.181.70"
    port = 3306
    user = "root"
    passwd = "bupt"
    db = "cnpc_local_k40"
    charset = "utf8"
    db = pymysql.connect(host=host, port=port, user=user, passwd=passwd, db=db, charset=charset)

    return db


def exe_sql(sql):

    db = getDBConnect()
    cursor = db.cursor()

    cursor.execute(sql)

    data_df = cursor.fetchall()
    # 获取连接对象的描述信息
    columnDes = cursor.description
    columnNames = [columnDes[i][0] for i in range(len(columnDes))]
    df = pd.DataFrame([list(i) for i in data_df], columns=columnNames)

    cursor.close()
    closeDBConnect(db)

    return df

# 带值的
# def exe_sql(sql, valus):
#     db = getDBConnect()
#     cursor = db.cursor()
#
#     cursor.execute(sql)
#
#     data_df = cursor.fetchall()
#     # 获取连接对象的描述信息
#     columnDes = cursor.description
#     columnNames = [columnDes[i][0] for i in range(len(columnDes))]
#     df = pd.DataFrame([list(i) for i in data_df], columns=columnNames)
#
#     closeDBConnect(db)
#
#     return df


def closeDBConnect(db):

    db.close