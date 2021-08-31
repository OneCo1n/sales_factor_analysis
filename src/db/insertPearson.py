
# coding: utf-8

import csv
from db.dateBase import *
#
# # create_tablesql = """CREATE TABLE 3DLabInfo (
# #                     UserName VARCHAR(20) NOT NULL,
# #                     Platform VARCHAR(10),
# #                     Craft VARCHAR(20),
# #                     OAM VARCHAR(20),
# #                     Location VARCHAR(10),
# #                     VoicecardType VARCHAR(20),
# #                     VoicecardSN VARCHAR(20),
# #                     intraIP VARCHAR(10),
# #                     PCInternet VARCHAR(20),
# #                     PCDNS VARCHAR(20),
# #                     SignalIP VARCHAR(20),
# #                     ATESipResource VARCHAR(50),
# #                     3vlanIP VARCHAR(20),
# #                     H248resource VARCHAR(50),
# #                     H248SigIP VARCHAR(20),
# #                     H248Numbers VARCHAR(50),
# #                     GatewayUserID VARCHAR(50),
# #                     IsdnAteNum VARCHAR(20),
# #                     Comments VARCHAR(50)
# #         )"""
# insert_tablesql = " INSERT INTO 3DLabInfo (UserName, \
#                 Platform, Craft, OAM, Location, VoicecardType, VoicecardSN, intraIP, \
#                 PCInternet, PCDNS, SignalIP, ATESipResource, 3vlanIP, H248resource, \
#                 H248SigIP, H248Numbers, GatewayUserID, IsdnAteNum, Comments) VALUES"
# # Open Mysql connect
# db = getDBConnect()
#
# cursor = db.cursor()
# # # Drop the table if existed.
# # cursor.execute("DROP TABLE IF EXISTS 3DLabInfo")
# # # Create sql table.
# # cursor.execute(create_tablesql)
# # Insert data from csv file.
#
# csv_file_path ="F:\\论文\\data\\df_all_pearson2.csv"
#
# csv_filename ="df_all_pearson2.csv"
# table_name = "pearson_auto"
#
#     # 打开csv文件
# file = open(csv_file_path, 'r', encoding='utf-8')
#     # 读取csv文件第一行字段名，创建表
# reader = file.readline()
# b = reader.split(',')
# colum = ''
# for a in b:
#     colum = colum + a + ' varchar(255),'
# colum = colum[:-1]
# # 编写sql，create_sql负责创建表，data_sql负责导入数据
# create_sql = 'create table if not exists ' + table_name + ' ' + '(' + colum + ')' + ' DEFAULT CHARSET=utf8'
# data_sql = "LOAD DATA LOCAL INFILE '%s' INTO TABLE %s FIELDS TERMINATED BY ',' LINES TERMINATED BY '\\r\\n' IGNORE 1 LINES" % (csv_filename, table_name)
#
#     # 使用数据库
#
#     # 设置编码格式
# cursor.execute('SET NAMES utf8;')
# cursor.execute('SET character_set_connection=utf8;')
#     # 执行create_sql，创建表
# cursor.execute(create_sql)
#     # 执行data_sql，导入数据
# #cursor.execute(data_sql)
# db.commit()
#     # 关闭连接
#
# cursor.close()
# db.close()



from sqlalchemy import create_engine
# create_engine('mysql+pymysql://用户名:密码@主机/库名?charset=utf8')
engine = create_engine('mysql+pymysql://root:bupt@10.112.181.70/cnpc_local_k40?charset=utf8')
table_commom=pd.read_csv(r'F:/论文/data/df_all_pearson2.csv')
#将数据写入sql
pd.io.sql.to_sql(table_commom,'pearson_source',con = engine ,if_exists = 'append',index="False")
