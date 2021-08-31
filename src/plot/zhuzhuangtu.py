# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from db.dateBase import *

def draw_bar(labels, quants):
    width = 0.4
    ind = np.linspace(0.5,0.8)
    # make a square figure
    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    # Bar Plot
    #ax.bar(ind - width / 2, quants, width, color='green')
    ax.bar(labels, quants, width, color='green')

    # Set the ticks on x-axis
    ax.set_xticks(ind)
    ax.set_xticklabels(labels)
    # labels
    ax.set_xlabel('影响因素')
    ax.set_ylabel('相关程度')
    # title
    ax.set_title('销量影响因素相关性', bbox={'facecolor': '0.8', 'pad':1 })
    plt.grid(True)
    plt.show()
    plt.savefig("bar.jpg")
    plt.close()


db = getDBConnect()
cursor = db.cursor()
sql = "select `pearson_source`.`discount_rate2` AS `折扣`,`pearson_source`.`number_station` AS `进站人数`," \
      "`pearson_source`.`number_store` AS `进店人数`,`pearson_source`.`plant_stars` AS `加油站星级`," \
      "`pearson_source`.`store_class` AS `便利店等级`,`pearson_source`.`building_area` AS `建筑面积`,`pearson_source`." \
      "`business_hall` AS `营业厅面积`,`pearson_source`.`paking_area` AS `paking_area`,`pearson_source`.`store_area` AS `store_area`," \
      "`pearson_source`.`plant_keyanliang_desc` AS `plant_keyanliang_desc`,`pearson_source`.`plant_asset_0001` AS `plant_asset_0001`," \
      "`pearson_source`.`plant_asset_0002` AS `plant_asset_0002`,`pearson_source`.`plant_asset_0004` AS `plant_asset_0004`," \
      "`pearson_source`.`plant_asset_0003` AS `plant_asset_0003`,`pearson_source`.`road_class_0003` AS `road_class_0003`," \
      "`pearson_source`.`road_class_0001` AS `road_class_0001`,`pearson_source`.`road_class_0004` AS `road_class_0004`," \
      "`pearson_source`.`road_class_0002` AS `road_class_0002`,`pearson_source`.`plant_class_code_0004` AS `plant_class_code_0004`," \
      "`pearson_source`.`plant_class_code_0002` AS `plant_class_code_0002`,`pearson_source`.`plant_class_code_0003` AS `plant_class_code_0003`," \
      "`pearson_source`.`plant_class_code_0001` AS `plant_class_code_0001`,`pearson_source`.`plant_location_class_010` AS `plant_location_class_010`," \
      "`pearson_source`.`plant_location_class_005` AS `plant_location_class_005`,`pearson_source`.`plant_location_class_007` AS `plant_location_class_007`," \
      "`pearson_source`.`plant_location_class_001` AS `plant_location_class_001`,`pearson_source`.`plant_location_class_009` AS `plant_location_class_009`," \
      "`pearson_source`.`plant_location_class_011` AS `plant_location_class_011`,`pearson_source`.`plant_location_class_003` AS `plant_location_class_003`," \
      "`pearson_source`.`plant_location_class_008` AS `plant_location_class_008`,`pearson_source`.`plant_location_class_004` AS `plant_location_class_004`," \
      "`pearson_source`.`plant_location_class_006` AS `plant_location_class_006`,`pearson_source`.`promotion_type_NP` AS `promotion_type_NP`," \
      "`pearson_source`.`promotion_type_C` AS `promotion_type_C`,`pearson_source`.`promotion_type_G` AS `promotion_type_G`,`pearson_source`.`promotion_type_O` AS `降价促销-直接降价`," \
      "`pearson_source`.`promotion_type_A` AS `降价促销-绝对值折扣`,`pearson_source`.`plant_type_desc_普通站` AS `plant_type_desc_普通站`," \
      "`pearson_source`.`promotion_type_BB04` AS `promotion_type_BB04`,`pearson_source`.`promotion_type_P` AS `promotion_type_P`," \
      "`pearson_source`.`promotion_type_BB06` AS `promotion_type_BB06`,`pearson_source`.`promotion_type_BB10` AS `promotion_type_BB10`," \
      "`pearson_source`.`promotion_type_BB20` AS `promotion_type_BB20`,`pearson_source`.`promotion_type_BB11` AS `promotion_type_BB11`," \
      "`pearson_source`.`plant_location_class_002` AS `plant_location_class_002`,`pearson_source`.`promotion_type_BB05` AS `promotion_type_BB05`," \
      "`pearson_source`.`promotion_type_BB02` AS `promotion_type_BB02`,`pearson_source`.`promotion_type_BB24` AS `promotion_type_BB24`," \
      "`pearson_source`.`promotion_type_Z001` AS `promotion_type_Z001`,`pearson_source`.`promotion_type_BB01` AS `promotion_type_BB01`," \
      "`pearson_source`.`promotion_type_BB31` AS `promotion_type_BB31`,`pearson_source`.`promotion_type_BB14` AS `promotion_type_BB14` from `pearson_source`" \
      "where  material ='70000969' and plant ='AK0F'"

cursor.execute(sql)


data = cursor.fetchall()
#获取连接对象的描述信息
columnDes = cursor.description
columnNames = [columnDes[i][0] for i in range(len(columnDes))]
df_data = pd.DataFrame([list(i) for i in data], columns=columnNames)
for col in df_data.columns:
    df_data[col].fillna(value=0, inplace=True)
print(df_data)
temp = df_data[0:1]
# for i in len(temp):
#     print(temp[i])
print(temp)
train_data = np.array(temp) #先将数据框转换为数组

tl = train_data[0]
print(tl)

quants = tl
labels = columnDes
print(quants)
cursor.close()
db.close()
draw_bar(labels, quants)
