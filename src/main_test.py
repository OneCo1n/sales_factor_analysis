from compnents.material_plant_pearson import *
from correlation_analysis.regression import *
from db.data_extraction import getPlantJoinNumofpeopleCode
from db.data_storage import *
import time


def anlalysisPeopleOfPlant(plants):
    time_start = time.time()
    print("[info : Analysis on Influencing Factors of number of people entering the station] 相关性分析")
    for plant in plants:
        try:
            #material = input('input the material id: ')

            print("[info : analysis] 分析油站：" + plant)
            # 进店人数相关性分析（pearson计算）
            #df_pearson, dfDsv = peopleOfPlantPearson(plant)
            df_pearson, dfDsv = allPeopleOfPlantPearson()
            print(dfDsv)
            print(df_pearson)

            #insert_table_pearson(material,df_pearson)

            #a,b,r,pearson = multipleLinearRegression(dfDsv)

            #insert_table_regression(material,a,b,r)
        except:
            continue
    # 将回归系数存入数据库
    print("[info : analysis] 相关性分析结束 耗时：%.3fmin" % ((time.time() - time_start)/60))


def anlalysisAllPeopleOfPlant():
    time_start = time.time()
    print("[info : Analysis on Influencing Factors of number of people entering the station] 相关性分析")
    #try:


    print("[info : analysis] 分析开始")
        # 进店人数相关性分析（pearson计算）
    df_pearson, dfDsv = allPeopleOfPlantPearson()
    print(dfDsv)
    print(df_pearson)
    outputpath='F:\论文\data\\test\\df_peopleOfPlant_pearson.csv'
    df_pearson.to_csv(outputpath,index=False,header=True)

        #insert_table_pearson(material,df_pearson)

        #a,b,r,pearson = multipleLinearRegression(dfDsv)

        #insert_table_regression(material,a,b,r)
    #except:
     #   print("")
    # 将回归系数存入数据库
    print("[info : analysis] 相关性分析结束 耗时：%.3fmin" % ((time.time() - time_start)/60))


def anlalysisTry(materials):
    time_start = time.time()
    print("所有商品相关性分析......")
    for material in materials:

        try:

            #material = input('input the material id: ')

            print("分析商品：" + material)
    # 单个商品在所有油站相关性分析（pearson计算）
            df_pearson, dfDsv = oneMaterialOnAllPlantPearson(material)
            #print(df_pearson)

# outputpath='F:\论文\data\\test\\df_one_pearson'+ material+'.csv'
# df_pearson.to_csv(outputpath,index=False,header=True)
# 将相关系数存入数据库

# db = getDBConnect()
            #insert_table_pearson(material,df_pearson)


            a,b,r,pearson = multipleLinearRegression(dfDsv)

            #insert_table_regression(material,a,b,r)
        except:
            continue
    # 将回归系数存入数据库
    print("所有相关性分析结束 耗时：%.3fmin" % ((time.time() - time_start)/60))


# #所有商品在所有油站相关性分析（pearson）
#
# allMaterialOnAllPlantPearson()


def anlalysisNoTry(materials):
    time_start = time.time()
    print("所有商品相关性分析......")
    for material in materials:
        #material = input('input the material id: ')

        print("分析商品：" + material)
    # 单个商品在所有油站相关性分析（pearson计算）
        df_pearson, dfDsv = oneMaterialOnAllPlantPearson(material)
        #print(df_pearson)
# 将相关系数存入数据库
        #insert_table_pearson(material,df_pearson)

        a,b,r,pearson = multipleLinearRegression(dfDsv)

        #insert_table_regression(material,a,b,r)

    # 将回归系数存入数据库
    print("所有相关性分析结束 耗时：%.3fmin" % ((time.time() - time_start)/60))



def totalInfoProcessTry(materials):
    time_start = time.time()
    print("[data : info process] 总表数据处理")

    list_except_material = []
    # 单独获取所有油站数据，避免重复查询
    plantInfo = getPlantJoinNumofpeopleCode()

    for material in materials:

        try:
            oneMaterialTotalInfo(material, plantInfo)

        except:
            list_except_material.append(material)
            continue

    print("[data : info process] 总表数据处理结束 耗时：%.3fmin" % ((time.time() - time_start)/60))
    print("insert exception material: ")
    print(list_except_material)
    f = open("F:\\论文\\data\\e_material.txt", "w")
    f.writelines(list_except_material)
    f.close()


def totalInfoProcessNoTry(materials):
    time_start = time.time()
    print("[data : info process] 总表数据处理")

    # 单独获取所有油站数据，避免重复查询
    plantInfo = getPlantJoinNumofpeopleCode()

    for material in materials:

        oneMaterialTotalInfo(material, plantInfo)

    print("[data : info process] 总表数据处理结束 耗时：%.3fmin" % ((time.time() - time_start) / 60))



#000000000070251989
#000000000070000579
#000000000000400102


#000000000070251989 武夷山
#000000000070042192 雀巢
#000000000070003387 脉动
#000000000070047411 中南海 (包) 13MG

#
#
# insert_table_material(materials)
#materials = ['000000000070319266', '000000000070319277', '000000000070319281', '000000000070319298', '000000000070319283', '000000000070319285', '000000000070319288', '000000000070247053', '000000000070320463', '000000000070306510', '000000000070319280', '000000000070319279', '000000000070319300', '000000000070247033', '000000000070201004', '000000000070157834', '000000000070010614', '000000000070247407', '000000000070255072', '000000000070261620', '000000000000404946', '000000000070261657', '000000000070261656']
#materials = [  '000000000070255072', '000000000070261620', '000000000000404946', '000000000070261657', '000000000070261656']

materials = ['000000000070251989']
#materials = getMaterialsId()

#print(materials)

anlalysisNoTry(materials)

#anlalysisAllPeopleOfPlant()
#totalInfoProcessNoTry(materials)
