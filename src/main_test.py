from compnents.material_plant_pearson import *
from correlation_analysis.regression import *
from db.data_storage import *
import time

#000000000070251989
#000000000070000579
#000000000000400102


# materials = getMaterialsId().material
# insert_table_material(materials)
# materials = ['000000000070251989']

materials = getMaterialsId()

print(materials)

time_start = time.time()
print("所有商品相关性分析......")


for material in materials:

    try:
    #material = input('input the material id: ')

        print("分析商品：" + material)
    # 单个商品在所有油站相关性分析（pearson计算）
        df_pearson, dfDsv = oneMaterialOnAllPlantPearson(material)
        print(df_pearson)

# outputpath='F:\论文\data\\test\\df_one_pearson'+ material+'.csv'
# df_pearson.to_csv(outputpath,index=False,header=True)
# 将相关系数存入数据库

# db = getDBConnect()
        insert_table_pearson(material,df_pearson)


        a,b,r,pearson = multipleLinearRegression(dfDsv)

        insert_table_regression(material,a,b,r)
    except:
        continue
    # 将回归系数存入数据库
print("所有相关性分析结束 耗时：%.3fmin" % ((time.time() - time_start)/60))


# #所有商品在所有油站相关性分析（pearson）
#
# allMaterialOnAllPlantPearson()