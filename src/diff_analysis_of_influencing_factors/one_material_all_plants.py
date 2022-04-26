from diff_analysis_of_influencing_factors.data_set.data_division import model_data_division
from diff_analysis_of_influencing_factors.train_model.gbrt.gbrt import do_analysis_use_gbrt
from diff_analysis_of_influencing_factors.train_model.lasso_lightGBM.esf_gbdt import \
    do_analysis_use_esf_gbdt
from diff_analysis_of_influencing_factors.train_model.lasso_xgboost.lasso_xgboost import do_analysis_use_lasso_xgboost
from diff_analysis_of_influencing_factors.train_model.lightGBM.lightGBM import use_lightGBM, do_analysis_use_lightGBM
from diff_analysis_of_influencing_factors.train_model.xgboost.xgboost import do_analysis_use_xgboost
from flaskTest.views.data_views.total_views import get_total_by_material, get_total_desc_by_material
from diff_analysis_of_influencing_factors.data_set.columns_name import exgColumnsName
from matplotlib import pyplot
from xgboost import XGBClassifier
from xgboost import plot_importance
#from data_preprocess.columnsExg import *
import lightgbm as lgb
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from chinese_calendar import is_workday, is_holiday
import numpy as np

# sns.set_style('white',{'font.sans-serif':['simhei','Arial']})
sns.set(style='whitegrid',palette='tab10')
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['font.family'] = 'SimHei'
matplotlib.rcParams['font.size'] = 10
matplotlib.rcParams['axes.unicode_minus']=False

plt.rcParams['font.sans-serif']=['Microsoft YaHei']

# 分析单商品在所有油站销量影响因素


# 首先获取所有相关数据


def do_analysis_one_material_all_plants(material, start_time, end_time):

    # 获取该商品的销量数据、油站信息、poi信息、促销信息等，用于模型训练（已编码）
    data = get_total_by_material(material, start_time, end_time)


    # data['date'] = pd.DatetimeIndex(data['date']).date
    # data['year'] = pd.DatetimeIndex(data['date']).year
    # data['month'] = pd.DatetimeIndex(data['date']).month
    # data['weekday'] = pd.DatetimeIndex(data['date']).weekday + 1
    # data['is_holiday'] = data['date'].apply(lambda x: is_holiday(x))
    # data['is_holiday'] = data['is_holiday'].astype('int')

    # 更改列名
    data = exgColumnsName(data)
    #data = data.drop(labels=None, axis=1, index=None, columns=['进站人数','进店人数'], inplace=False)
    print(data.dtypes)
    print(data.describe().T)
    #print(np.isnan(data).any())

    # fig, axes_q = plt.subplots(1, 2, 1)
    # data['quantity'].plot(kind='box', ax=axes_q)
    # axes_q.set_xlabel('日销量')
    # axes_q.set_ylabel('')

   #  plt.subplot(1, 2, 1)
   #  data['quantity'].plot(kind='box')
   #  plt.xlabel('日销量')
   #  plt.savefig('销量箱型图.png', bbox_inches='tight')
   # # plt.set_xlabel('日销量')
   #  #plt.set_ylabel('')
   #
   #  plt.subplot(1, 2, 2)
   #  data['折扣率'].plot(kind='box')
   #  plt.savefig('折扣率箱型图.png', bbox_inches='tight')
   # # plt.set_xlabel('')
   # # plt.set_ylabel('')
   #  plt.show()

    # 划分数据集
    # 对销量进行分析
    target_name = 'quantity'

    # mse, rmse, r2, mae, mape, importance_dict = do_analysis_use_lightGBM(data, target_name)
   # mse, rmse, r2, importance_dict = do_analysis_use_lasso_xgboost(data, target_name)
    #mse, rmse, r2, mae, mape, importance_dict = do_analysis_use_esf_gbdt(data, target_name)
    # mse, rmse, r2, mae, mape, importance_dict = do_analysis_use_xgboost(data, target_name)
    mse, rmse, r2, mae, mape, importance_dict = do_analysis_use_gbrt(data, target_name)


    print(importance_dict)
    plot_importance(importance_dict)
    pyplot.show()
    #
    # # 获取该商品的销量描述信息、油站描述信息、poi信息等（未编码）展示上面得出的影响因素与销量的具体影响
    # data_desc = get_total_desc_by_material(material,start_time, end_time)
    # # 更改列名
    # data_desc = exgColumnsName(data_desc)
    #
    #
    # g1 = sns.pairplot(data_desc,
    #              height=4,
    #              x_vars=['油站类别'],
    #              y_vars=['quantity'], plot_kws={'alpha': 0.1})
    # g1.fig.set_size_inches(25, 5)
    # g1.savefig('单油品油站类别.png', bbox_inches='tight')
    #
    # g2 = sns.pairplot(data_desc,
    #              height=4,
    #              x_vars=['加油站位置'],
    #              y_vars=['quantity'], plot_kws={'alpha': 0.1})
    # g2.fig.set_size_inches(15, 5)
    # g2.savefig('单油品油站位置.png', bbox_inches='tight')
    #
    # # g3 = sns.pairplot(data_desc,
    # #                   height=4,
    # #                   x_vars=['道路等级'],
    # #                   y_vars=['quantity'], plot_kws={'alpha': 0.1})
    # # #g3.fig.set_size_inches(30, 5)
    # # g3.savefig('单油品道路等级.png', bbox_inches='tight')
    #
    # plt.show()
    #

    # 通过上面的特征重要度dict 来进行字段筛选



    pass
# 000000000070251989
# 000000000070042192
# 000000000070003387
# 000000000070047411
do_analysis_one_material_all_plants('000000000070251989', '2019-09-01' , '2019-12-31')