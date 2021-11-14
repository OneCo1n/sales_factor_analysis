import matplotlib

from compnents.material_plant_pearson import allPeopleOfPlantPearson
from data_preprocess.columnsExg import *
from db.dataQuery import *
from pandas import Series,DataFrame
from numpy.random import randn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import matplotlib as mpl
from model.useXgboost import *

# font_set = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=15)
#设置字体为楷体 可解决中文乱码问题
#matplotlib.rcParams['font.sans-serif'] = ['KaiTi']
plt.rcParams['font.sans-serif']=['Microsoft YaHei']



#000000000070251989
#000000000070000579
#000000000000400102


def num2color(values, cmap):
    """将数值映射为颜色"""
    norm = mpl.colors.Normalize(vmin=np.min(values), vmax=np.max(values))
    cmap = mpl.cm.get_cmap(cmap)
    return [cmap(norm(val)) for val in values]


def displayPearson(material):

    # 获取商品销量影响因素相关系数
    df_pearson = getMaterialsPearson(material)
    # 获取商品名称
    name = getMaterialName(material)
    print(df_pearson)
    df_pearson = df_pearson.drop(labels=None, axis=1, index=None, columns=['material','plant_type_desc'], inplace=False)
    # 将影响因素修改为中文
    df_pearson = exgColumnsName(df_pearson)
    df_pearson = df_pearson.astype(float)

    # c_p = np.arange(18, -15, -1)
    # colors_positive = num2color(c_p, "rainbow")
    # c_n = np.arange(-15, 18, -1)
    # colors_negative = num2color(c_n, "rainbow")
    c = np.arange(33, 0, -3)
    colors = num2color(c, "rainbow")

    #colors = np.where(np.array(df_pearson.iloc[0]) > 0, colors_positive, colors_negative)
    df_pearson.plot(kind='bar' , grid = True, title = name + u'销量相关性分析', color =colors)

    plt.show()


def displayPearsonByCategory(category):

    # 获取商品销量影响因素相关系数
    df_pearson = getMaterialsPearsonByBigCategory(category)
    # 获取商品名称
    name = getBigCategoryName(category)
    print(df_pearson)
    df_pearson = df_pearson.drop(labels=None, axis=1, index=None, columns=['big_category'], inplace=False)
    # 将影响因素修改为中文
    df_pearson = exgColumnsName(df_pearson)
    df_pearson = df_pearson.astype(float)

    # c_p = np.arange(18, -15, -1)
    # colors_positive = num2color(c_p, "rainbow")
    # c_n = np.arange(-15, 18, -1)
    # colors_negative = num2color(c_n, "rainbow")
    c = np.arange(33, 0, -3)
    colors = num2color(c, "rainbow")

    #colors = np.where(np.array(df_pearson.iloc[0]) > 0, colors_positive, colors_negative)
    df_pearson.plot(kind='bar' , grid = True, title = name + u'销量相关性分析', color =colors)

    plt.show()

def displayPeopleOfPlantPearson():

    # 获取进店人数影响因素相关系数
    df_pearson, dfDsv = allPeopleOfPlantPearson()

    # df_pearson = df_pearson.drop(labels=None, axis=1, index=None, columns=['material','plant_type_desc'], inplace=False)

    df_pearson = pd.DataFrame(df_pearson)
    df_number_station = pd.DataFrame(df_pearson[0:1])
    df_number_station = df_number_station.drop(labels=None, axis=1, index=None, columns=['number_station','number_store'],
                                               inplace=False)

    # 将影响因素修改为中文
    df_number_station = exgColumnsName2(df_number_station)
    df_number_station = df_number_station.astype(float)


    # c_p = np.arange(18, -15, -1)
    # colors_positive = num2color(c_p, "rainbow")
    # c_n = np.arange(-15, 18, -1)
    # colors_negative = num2color(c_n, "rainbow")
    c = np.arange(33, 0, -3)
    colors = num2color(c, "rainbow")

    #colors = np.where(np.array(df_pearson.iloc[0]) > 0, colors_positive, colors_negative)
    df_number_station.plot(kind='bar' , grid = True, title = u'进站人数影响因素相关性分析', color =colors, rot = 90)
    plt.show()

def displayPearsonByBigCategory(bigCategory):
    # 获取商品销量影响因素相关系数
    df_pearson = getMaterialsPearsonByBigCategory(bigCategory)
    # 获取商品名称
    name = getBigCategoryName(bigCategory)
    print(df_pearson)
    df_pearson = df_pearson.drop(labels=None, axis=1, index=None, columns=['big_category'], inplace=False)
    # 将影响因素修改为中文
    df_pearson = exgColumnsName(df_pearson)
    df_pearson = df_pearson.astype(float)

    # c_p = np.arange(18, -15, -1)
    # colors_positive = num2color(c_p, "rainbow")
    # c_n = np.arange(-15, 18, -1)
    # colors_negative = num2color(c_n, "rainbow")
    c = np.arange(33, 0, -3)
    colors = num2color(c, "rainbow")

    # colors = np.where(np.array(df_pearson.iloc[0]) > 0, colors_positive, colors_negative)
    df_pearson.plot(kind='bar', grid=True, title=name + u'销量相关性分析', color=colors)

    plt.show()


def displayRegression(material):

    # 获取商品销量影响因素相关系数
    df_regression, r = getMaterialsRegression(material)
    # 获取商品名称
    name = getMaterialName(material)
    print(df_regression)
    #df_pearson = df_regression.drop(labels=None, axis=1, index=None, columns=['material','plant_type_desc'], inplace=False)
    # 将影响因素修改为中文
    df_regression = exgColumnsName(df_regression)
    df_regression = df_regression.astype(float)

    # c_p = np.arange(18, -15, -1)
    # colors_positive = num2color(c_p, "rainbow")
    # c_n = np.arange(-15, 18, -1)
    # colors_negative = num2color(c_n, "rainbow")
    c = np.arange(33, 0, -3)
    colors = num2color(c, "rainbow")

    #colors = np.where(np.array(df_pearson.iloc[0]) > 0, colors_positive, colors_negative)
    df_regression.plot(kind='bar' , grid = True,title = name + u'销量影响因素影响程度', color =colors)

    plt.show()


# material = input('input material: ')
#
# # 展示相关性 pearson系数
# displayPearson("000000000070251989")
#
# # 特征贡献度
# trainMaterialXgboost(material)
#
# # 重要程度
# displayRegression(material)
#
# # 模型准确率

#displayPeopleOfPlantPearson()

displayPearsonByBigCategory("5001")
