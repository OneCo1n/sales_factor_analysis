# coding=utf-8
from __future__ import division
import numpy as np
import pandas as pd
import random
import math
from sklearn import metrics
from sklearn.model_selection import train_test_split
import xgboost as xgb

import lightgbm as lgb
from random import randint

from diff_analysis_of_influencing_factors.data_set.data_division import model_data_division
from diff_analysis_of_influencing_factors.train_model.lightGBM.lightGBM import use_lightGBM, do_analysis_use_lightGBM
from flaskTest.views.data_views.total_views import get_total_by_material, get_total_desc_by_material, \
    get_total_by_material_and_company, get_total_desc_by_material_and_company
from diff_analysis_of_influencing_factors.data_set.columns_name import exgColumnsName
from matplotlib import pyplot
from xgboost import XGBClassifier
from xgboost import plot_importance
import pandas as pd
from chinese_calendar import is_workday, is_holiday
# from data_preprocess.columnsExg import *
import lightgbm as lgb
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

# from xgboost.sklearn import XGBClassifiers


'''
群体大小，一般取20~100；终止进化代数，一般取100~500；交叉概率，一般取0.4~0.99；变异概率，一般取0.0001~0.1。
'''
# generations = 400   # 繁殖代数 100
pop_size = 500  # 种群数量  500
# max_value = 10      # 基因中允许出现的最大值 （可防止离散变量数目达不到2的幂的情况出现，限制最大值，此处不用）
chrom_length = 15  # 染色体长度
pc = 0.6  # 交配概率
pm = 0.01  # 变异概率
results = []  # 存储每一代的最优解，N个三元组（auc最高值, n_estimators, max_depth）
fit_value = []  # 个体适应度
fit_mean = []  # 平均适应度
# pop = [[0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0] for i in range(pop_size)] # 初始化种群中所有个体的基因初始序列

random_seed = 20
cons_value = 0.19 / 31  # (0.20-0.01）/ (32 - 1)

'''要调试的参数有：（参考：http://xgboost.readthedocs.io/en/latest/parameter.html）
   tree_num：基树的棵数   ----------------（要调的参数）
   eta: 学习率（learning_rate），默认值为0.3，范围[0,1]  ----------------（要调的参数）
   max_depth: 最大树深，默认值为6   ----------------（要调的参数）
   min_child_weight：默认值为1，范围[0, 正无穷]，该参数值越小，越容易 overfitting，当它的值较大时，可以避免模型学习到局部的特殊样本。 ----------（要调的参数）
   gamma：默认值为0，min_split_loss，范围[0, 正无穷]
   subsample：选择数据集百分之多少来训练，可以防止过拟合。默认值1，范围(0, 1]，理想值0.8
   colsample_bytree：subsample ratio of columns when constructing each tree，默认值1，范围(0, 1]，理想值0.8，太小的值会造成欠拟合
   lambda：L2 regularization term on weights, increase this value will make model more conservative.参数值越大，模型越不容易过拟合
   alpha：L1 regularization term on weights, increase this value will make model more conservative.参数值越大，模型越不容易过拟合
   上述参数，要调的有4个，其他的采用理想值就可以
   tree_num: [10、 20、 30、......150、160] 用4位二进制, 0000代表10
   eta: [0.01, 0.02, 0.03, 0.04, 0.05, ...... 0.19, 0.20]   0.2/0.01=20份，用5位二进制表示足够（2的4次方<20<2的5次方）
       00000 -----> 0.01
       11111 -----> 0.20
       0.01 + 对应十进制*（0.20-0.01）/ (2的5次方-1)
   max_depth:[3、4、5、6、7、8、9、10]   用3位二进制
   min_child_weight: [1, 2, 3, 4, 5, 6, 7, 8]  用3位二进制
   示例：   0010,         01001,               010,      110  （共15位）
         tree_num         eta               max_depth  min_child_weight
        (1+2)*10=30  0.01+9*0.005939=0.06       3+2=5      1+6=7
'''


# 定义评价函数rmspe    均方根对数误差
def rmspe(y, yhat):
    return np.sqrt(np.mean((yhat / y - 1) ** 2))


def lightgbmModel(tree_num, eta, max_depth, min_child_weight, random_seed,  X_train, X_test, y_train, y_test):
    # ---------------------------数据准备------------------------
    # data = pd.read_csv('UnFeature.csv', low_memory=False)
    # # 因为销售额为0的记录不计入评分，所以只采用店铺为开，且销售额大于0的数据进行训练
    # data = data[data["Open"] != 0]
    # data = data[data["Sale"] > 0]
    #
    # data = data[:500000]
    #
    # data.fillna(0, inplace=True)
    #
    # # 不带特征工程
    # data.drop(['Date', 'Open', 'PromoInterval', 'monthStr', 'Customers'], axis=1, inplace=True)
    #
    # # 不带特征工程
    #
    # # data.drop(['Date','Customers','Open','PromoInterval','monthStr'],axis=1,inplace =True)
    # target = np.log1p(data.Sale)
    # data.drop(['Sale'], axis=1, inplace=True)
    # X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2)
    # --------------------------------------------------------------
    # ---------------------------数据准备------------------------
    # data = get_total_by_material_and_company('000000000070251989', '2019-01-01', '2019-12-31', '北京销售公司二分公司')
    # #
    # # data['date'] = pd.DatetimeIndex(data['date']).date
    # # data['year'] = pd.DatetimeIndex(data['date']).year
    # # data['month'] = pd.DatetimeIndex(data['date']).month
    # # data['weekday'] = pd.DatetimeIndex(data['date']).weekday + 1
    # # data['is_holiday'] = data['date'].apply(lambda x: is_holiday(x))
    # # data['is_holiday'] = data['is_holiday'].astype('int')
    #
    # # 更改列名
    # data = exgColumnsName(data)
    #
    # # 划分数据集
    # # 对销量进行分析
    # target_name = 'quantity'
    # X_train, X_test, y_train, y_test = model_data_division(data, target_name, 0.2)

    # --------------------------------------------------------------

    # 创建成lgb特征的数据集格式
    lgb_train = lgb.Dataset(X_train, y_train)  # 将数据保存到LightGBM二进制文件将使加载更快
    lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)  # 创建验证数据
    params = {
        'task': 'train',
        'boosting_type': 'gbdt',  # 设置提升类型
        'objective': 'regression',  # 目标函数
        'early_stopping_rounds': 100,
        'eval_metric': 'rmse',  # 评估函数
        # 'num_leaves': 31,   # 叶子节点数
        'learning_rate': eta,  # 学习速率
        'feature_fraction': 0.8,  # 建树的特征选择比例
        'bagging_fraction': 0.8,  # 建树的样本采样比例
        # 'bagging_freq': 5,  # k 意味着每 k 次迭代执行bagging
        'verbose': 1,  # <0 显示致命的, =0 显示错误 (警告), >0 显示信息
        'max_depth': max_depth,
        'min_child_weight': min_child_weight,
        'seed': randint(1, 10)
    }
    print('Start training...')
    # 训练 cv and train
    model = lgb.train(params, lgb_train, num_boost_round=tree_num, valid_sets=lgb_eval)  # 训练数据需要参数列表和数据集
    yhat = model.predict(X_test, num_iteration=model.best_iteration)
    error = rmspe(np.expm1(yhat), np.expm1(y_test))

    return error


def loadFile(filePath):
    fileData = pd.read_csv(filePath)
    return fileData


# Step 1 : 对参数进行编码（用于初始化基因序列，可以选择初始化基因序列，本函数省略）
def geneEncoding(pop_size, chrom_length):
    pop = [[]]
    for i in range(pop_size):
        temp = []
        for j in range(chrom_length):
            temp.append(random.randint(0, 1))
        pop.append(temp)
    return pop[1:]


# Step 2 : 计算个体的目标函数值
def cal_obj_value(pop,  X_train, X_test, y_train, y_test):
    objvalue = []
    variable = decodechrom(pop)
    for i in range(len(variable)):
        tempVar = variable[i]

        tree_num_value = (tempVar[0] + 1) * 70  # *10
        eta_value = 0.01 + tempVar[1] * cons_value
        max_depth_value = 3 + tempVar[2]
        min_child_weight_value = 1 + tempVar[3]

        # aucValue = xgboostModel(tree_num_value, eta_value, max_depth_value, min_child_weight_value, random_seed)
        error = lightgbmModel(tree_num_value, eta_value, max_depth_value, min_child_weight_value, random_seed,  X_train, X_test, y_train, y_test)

        # objvalue.append(aucValue)
        objvalue.append(error)
    return objvalue  # 目标函数值objvalue[m] 与个体基因 pop[m] 对应


# 对每个个体进行解码，并拆分成单个变量，返回 tree_num（4）、eta（5）、max_depth（3）、min_child_weight（3）
def decodechrom(pop):
    variable = []
    for i in range(len(pop)):
        res = []

        # 计算第一个变量值，即 0101->10(逆转)
        temp1 = pop[i][0:4]
        v1 = 0
        for i1 in range(4):
            v1 += temp1[i1] * (math.pow(2, i1))
        res.append(int(v1))

        # 计算第二个变量值
        temp2 = pop[i][4:9]
        v2 = 0
        for i2 in range(5):
            v2 += temp2[i2] * (math.pow(2, i2))
        res.append(int(v2))

        # 计算第三个变量值
        temp3 = pop[i][9:12]
        v3 = 0
        for i3 in range(3):
            v3 += temp3[i3] * (math.pow(2, i3))
        res.append(int(v3))

        # 计算第四个变量值
        temp4 = pop[i][12:15]
        v4 = 0
        for i4 in range(3):
            v4 += temp4[i4] * (math.pow(2, i4))
        res.append(int(v4))

        variable.append(res)
    return variable


# Step 3: 计算个体的适应值（计算最大值，于是就淘汰负值就好了）
def calfitvalue(obj_value):
    fit_value = []
    temp = 1.0  # 0.0
    Cmin = 0
    for i in range(len(obj_value)):
        if (obj_value[i] + Cmin < 1):  # >0
            temp = Cmin + obj_value[i]
        else:
            temp = 1.0
        fit_value.append(temp)
    return fit_value


# Step 4: 找出适应函数值中最大值，和对应的个体
def best(pop, fit_value):
    best_individual = pop[0]
    best_fit = fit_value[0]
    for i in range(1, len(pop)):
        # if(fit_value[i] > best_fit):
        if (fit_value[i] < best_fit):
            best_fit = fit_value[i]
            best_individual = pop[i]
    return [best_individual, best_fit]


# Step 5: 每次繁殖，将最好的结果记录下来(将二进制转化为十进制)
def b2d(best_individual):
    # 计算第一个变量值
    temp1 = best_individual[0:4]
    v1 = 0
    for i1 in range(4):
        v1 += temp1[i1] * (math.pow(2, i1))
    v1 = (v1 + 1) * 70  # *10

    # 计算第二个变量值
    temp2 = best_individual[4:9]
    v2 = 0
    for i2 in range(5):
        v2 += temp2[i2] * (math.pow(2, i2))
    v2 = 0.01 + v2 * cons_value

    # 计算第三个变量值
    temp3 = best_individual[9:12]
    v3 = 0
    for i3 in range(3):
        v3 += temp3[i3] * (math.pow(2, i3))
    v3 = 3 + v3

    # 计算第四个变量值
    temp4 = best_individual[12:15]
    v4 = 0
    for i4 in range(3):
        v4 += temp4[i4] * (math.pow(2, i4))
    v4 = 1 + v4

    return int(v1), float(v2), int(v3), int(v4)


# Step 6: 自然选择（轮盘赌算法）
def selection(pop, fit_value):
    # 计算每个适应值的概率
    new_fit_value = []
    total_fit = sum(fit_value)
    for i in range(len(fit_value)):
        new_fit_value.append(fit_value[i] / total_fit)
    # 计算每个适应值的累积概率
    cumsum(new_fit_value)
    # 生成随机浮点数序列
    ms = []
    pop_len = len(pop)
    for i in range(pop_len):
        ms.append(random.random())
    # 对生成的随机浮点数序列进行排序
    ms.sort()
    # 轮盘赌算法（选中的个体成为下一轮，没有被选中的直接淘汰，被选中的个体代替）
    fitin = 0
    newin = 0
    newpop = pop
    while newin < pop_len:
        if (ms[newin] < new_fit_value[fitin]):
            newpop[newin] = pop[fitin]
            newin = newin + 1
        else:
            fitin = fitin + 1
    pop = newpop


# 求适应值的总和
def sum(fit_value):
    total = 0
    for i in range(len(fit_value)):
        total += fit_value[i]
    return total


# 计算累积概率
def cumsum(fit_value):
    temp = []
    for i in range(len(fit_value)):
        t = 0
        j = 0
        while (j <= i):
            t += fit_value[j]
            j = j + 1
        temp.append(t)
    for i in range(len(fit_value)):
        fit_value[i] = temp[i]


# Step 7: 交叉繁殖
def crossover(pop, pc):  # 个体间交叉，实现基因交换
    poplen = len(pop)
    for i in range(poplen - 1):
        if (random.random() < pc):
            cpoint = random.randint(0, len(pop[0]))
            temp1 = []
            temp2 = []
            temp1.extend(pop[i][0: cpoint])
            temp1.extend(pop[i + 1][cpoint: len(pop[i])])
            temp2.extend(pop[i + 1][0: cpoint])
            temp2.extend(pop[i][cpoint: len(pop[i])])
            pop[i] = temp1
            pop[i + 1] = temp2


# Step 8: 基因突变
def mutation(pop, pm):
    px = len(pop)
    py = len(pop[0])
    for i in range(px):
        if (random.random() < pm):
            mpoint = random.randint(0, py - 1)
            if (pop[i][mpoint] == 1):
                pop[i][mpoint] = 0
            else:
                pop[i][mpoint] = 1


def writeToFile(var, w_path):
    f = open(w_path, "a+")
    for item in var:
        f.write(str(item) + "\r\n")
    f.close()


def generAlgo(generations,  X_train, X_test, y_train, y_test):
    pop = geneEncoding(pop_size, chrom_length)
    print(str(generations) + " start...")
    for i in range(generations):
        # print("第 " + str(i) + " 代开始繁殖......")
        obj_value = cal_obj_value(pop,  X_train, X_test, y_train, y_test)  # 计算目标函数值
        # print(obj_value)
        fit_value = calfitvalue(obj_value)  # 计算个体的适应值
        # print(fit_value)
        [best_individual, best_fit] = best(pop, fit_value)  # 选出最好的个体和最好的函数值
        # print("best_individual: "+ str(best_individual))
        v1, v2, v3, v4 = b2d(best_individual)
        results.append([best_fit, v1, v2, v3, v4])  # 每次繁殖，将最好的结果记录下来
        # print(str(best_individual) + " " + str(best_fit))
        selection(pop, fit_value)  # 自然选择，淘汰掉一部分适应性低的个体
        crossover(pop, pc)  # 交叉繁殖
        mutation(pop, pc)  # 基因突变
    results.sort()

    writeToFile(results, "generation_" + str(generations) + ".txt")
    print(results[-1])


if __name__ == '__main__':
    gen = [10]  # gen = [100, 200, 300, 400, 500]
    data = get_total_by_material_and_company('000000000070251989', '2019-01-01', '2019-12-31', '北京销售公司二分公司')
    data = exgColumnsName(data)
    target_name = 'quantity'
    X_train, X_test, y_train, y_test = model_data_division(data, target_name, 0.2)

    for g in gen:
        generAlgo(int(g),  X_train, X_test, y_train, y_test)
