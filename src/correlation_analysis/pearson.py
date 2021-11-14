import pandas as pd
import matplotlib.pyplot as plt  # 可视化
import seaborn as sns  # 可视化
from data_preprocess.dtype_exchange import *
from data_division.division import divisionByPlant
import time


def pearson_one_plant(df):
    #df = pd.read_csv('boston_housing.csv')  # 读取数据

    # print('pearson\n', df.corr(method='pearson'))  # 皮尔逊相关系数
    # print('kendall\n', df.corr(method='kendall'))  # 肯德尔秩相关系数
    # print('spearman\n', df.corr(method='spearman'))  # 斯皮尔曼秩相关系数
    del_features=['']
    #df.drop()
    # print(df.dtypes)

    # _, ax = plt.subplots(figsize=(15, 10))  # 分辨率1200×1000
    corr = df.corr(method='pearson')  # 使用皮尔逊系数计算列与列的相关性
    return corr
    #corr = corr[]
    # print(corr)
    #
    # outputpath='F:\论文\data\\df2_ps_1.csv'
    # corr.to_csv(outputpath,index=False,header=True)
    # # j'j'a
    # # corr = df.corr(method='kendall')
    # # corr = df.corr(method='spearman')
    # cmap = sns.diverging_palette(220, 10, as_cmap=True)  # 在两种HUSL颜色之间制作不同的调色板。图的正负色彩范围为220、10，结果为真则返回matplotlib的colormap对象
    # _ = sns.heatmap(
    #     corr,  # 使用Pandas DataFrame数据，索引/列信息用于标记列和行
    #     cmap=cmap,  # 数据值到颜色空间的映射
    #     square=True,  # 每个单元格都是正方形
    #     cbar_kws={'shrink': .9},  # `fig.colorbar`的关键字参数
    #     ax=ax,  # 绘制图的轴
    #     annot=True,  # 在单元格中标注数据值
    #     annot_kws={'fontsize': 12})  # 热图，将矩形数据绘制为颜色编码矩阵
    # plt.show()

def pearson_all_plant(df, material):

    #建立pearson的dataframe

    # cor = pd.DataFrame
    time_start = time.time()
    print("正在进行%s销量的相关性分析......" % material)

    plant_class = df['plant'].unique()

    create_flag = False

    for plant in plant_class:

        df_one_plant = divisionByPlant(df, plant)
        corr_one: pd.DataFrame  = pearson_one_plant(df_one_plant)


        if(create_flag == False):
            #构建pearson列名
            create_col = corr_one.columns.insert(0,'plant')
            create_col = create_col.insert(0, 'material')
            global cor
            cor = pd.DataFrame(columns=(create_col))
           # cor = pearsonDytypeExg(cor)
            create_flag = True

        # 取每个油站pearson结果的第一行，只取其他特征与销量的pearson系数，插入到总表中
        temp_row = corr_one[0:1]
        #重置索引列
        temp_row = temp_row.reset_index(drop=True)
        temp_row.insert(0, 'plant', plant, allow_duplicates=False)
        temp_row.insert(0, 'material', material, allow_duplicates=False)


        # print(cor)
        # print(temp_row)
        #一个油站的数据当作一条数据插入总表
        cor = cor.append(temp_row, ignore_index=True)

    # outputpath='F:\论文\data\\df2_ps_3.csv'
    # cor.to_csv(outputpath,index=True,header=True)
    print("商品%s相关性分析结束 耗时：%.3fs" %( material, (time.time() - time_start)))

    return cor

# # 使用方法是DataFrame.insert(loc, column, value, allow_duplicates=False)
# # 即df.insert(添加列位置索引序号，添加列名，数值，是否允许列名重复)
# df.insert(1, 'tail', 1, allow_duplicates=False)
# df
#   name  tail  number  leg  head
# 0  cat     1       3   12     1



