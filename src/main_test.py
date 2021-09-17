from compnents.material_plant_pearson import *

#000000000070251989
#000000000070000579
#000000000000400102


material = input('input the material id: ')

# 单个商品在所有油站相关性分析（pearson计算）
df_pearson = oneMaterialOnAllPlantPearson(material)

outputpath='F:\论文\data\\test\\df_one_pearson'+ material+'.csv'
df_pearson.to_csv(outputpath,index=False,header=True)

#将数据插入到数据库表中
#insert_table_pearson(df_pearson)



#所有商品在所有油站相关性分析（pearson）

#allMaterialOnAllPlantPearson()