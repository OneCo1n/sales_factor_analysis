from db import data_extraction
from data_encoding.encoder import *
from correlation_analysis.pearson import *
from db.data_storage import insert_table_pearson
from factor_analysis.factor_analysis import *
from db.dataQuery import *
from data_preprocess.standardizing import *
from data_preprocess.outlierProcessing import *
from plot.pairPlot import *

#
# True_P=0.5
#
# def sampling(N):
#     ## 产生Bernouli样本
#     x=nprd.rand(N)<True_P
#     return x
#
# M=10000 #模拟次数
# xbar=np.zeros(M)
# N=np.array([i+1 for i in range(M)])
# x=sampling(M)
# for i in range(M):
#     if i==0:
#         xbar[i]=x[i]
#     else:
#         xbar[i]=(x[i]+xbar[i-1]*i)/(i+1)
#
# ## 导入matplotlib
# import matplotlib.pyplot as plt
# ## 使图形直接插入到jupyter中
# ##%matplotlib inline
# # 设定图像大小
# plt.rcParams['figure.figsize'] = (10.0, 8.0)
#
# plt.plot(N,xbar,label=r'$\bar{x}$',color='pink') ## xbar
# xtrue=np.ones(M)*True_P
# plt.plot(N,xtrue,label=r'$0.5$',color='black') ## true xbar
# plt.xlabel('N')
# plt.ylabel(r'$\bar{x}$')
# plt.legend(loc='upper right', frameon=True)
# plt.show() ## 画图




def oneMaterialOnAllPlantPearson(material):
    #从数据库读取数据 并进行预处理：数据清洗、数据转换、缺失值处理、数据合并
    df = data_extraction.get_df_from_db(material)
    # print(df.dtypes)

    # df.boxplot()
    # plt.savefig("boxplot_" + material +".jpg")
    # plt.show()
    # 离群值处理
    outlierProcessing(df)


    # df.boxplot()
    # plt.savefig("boxplot_after_" + material + ".jpg")
    # plt.show()

    #对数据进行序列编码、独热编码
    df = encoding(df)


    # 对数据进行标准化
    # print(df.dtypes)
    df = pearsonDytypeExg(df)
    minMaxScaler(df)

    # 销量分析所需要的数据
    # dfDsv = df.drop(labels=None, axis=1, index=None, columns=['plant','date','plant_type_desc'], inplace=False)

    # 进站人数影响因素分析需要的数据
    dfDsv = df.drop(labels=None, axis=1, index=None, columns=['plant', 'quantity' ,'date', 'plant_type_desc'], inplace=False)



    # outputpath = 'F:\论文\data\\test\\bill_' + material + '.csv'
    # dfDsv.to_csv(outputpath, index=False, header=True)

    # 单变量散点图
    # pairplot(df, material)


    # print(df)
    # print(df.dtypes)
    #factorAnalysis(df)

    #销售商品的油站
    # print(df['plant'].unique())

    #计算所有油站单个商品pearson系数
    # df_pearson = pearson_all_plant(df, material)
    df_pearson = df.corr()

    return  df_pearson,dfDsv
    #return df



def allMaterialOnAllPlantPearson():
    time_start = time.time()
    print("----------商品销量相关性分析开始----------")
    #获取所有商品列表
    materials_df = getMaterialsId()
    # print(materials_df)
    materials = materials_df['material']
    print(materials)
    create_flag = False

    #遍历商品列表，计算商品列表中商品的相关系数
    for material in materials:
        #单个商品数据
        df_pearson_oneMaterial = oneMaterialOnAllPlantPearson(material)

        # 创建总表
        if (create_flag == False):
            df_pearson = pd.DataFrame(columns=(df_pearson_oneMaterial.columns))
            create_flag = True

        #单表插入到总表中
        df_pearson = df_pearson.append(df_pearson_oneMaterial, ignore_index=True)

        outputpath='F:\论文\data\\df_all_pearson.csv'
        df_pearson.to_csv(outputpath,index=False,header=True)

    print("商品相关性分析结束 耗时：%.3fs" % (time.time() - time_start))
    return  df_pearson






# outputpath='F:\论文\data\\df.csv'
# df.to_csv(outputpath,index=False,header=True)

# outputpath='F:\论文\data\\df_encoding.csv'
# df.to_csv(outputpath,index=False,header=True)

# df_temp = divisionByPlant(df,'AB01')
#
# #箱型图
# drawBox(df_temp)
# #散点图
# drawScatter(df_temp,'discount_rate2', 'quantity')
# #曲线图
# sales_plots_byPlant(df, 'AB01')
# #sales_plots_allPlant(df)

#pearson_one_plant(df_temp)

# correlation_analysis(df_AB01)





#outputpath='F:\论文\data\\df2_ps.csv'
#df.to_csv(outputpath,index=False,header=True)
# #数据预
# #特征工程
# #相关性分析
# correlation_analysis.correlation_analysis(df)








