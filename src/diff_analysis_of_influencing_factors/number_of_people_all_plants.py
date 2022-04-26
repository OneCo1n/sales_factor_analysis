import time

from diff_analysis_of_influencing_factors.data_set.data_division import model_data_division
from diff_analysis_of_influencing_factors.train_model.gbrt.gbrt import do_analysis_use_gbrt
from diff_analysis_of_influencing_factors.train_model.lasso_lightGBM.esf_gbdt import \
    do_analysis_use_esf_gbdt
from diff_analysis_of_influencing_factors.train_model.lightGBM.lightGBM import use_lightGBM, do_analysis_use_lightGBM
from diff_analysis_of_influencing_factors.train_model.xgboost.xgboost import do_analysis_use_xgboost
from flaskTest.preprocess.bill.bill_preprocess import bill_log
from flaskTest.views.data_views.total_views import get_total_by_material, get_total_desc_by_material, get_total_people, \
    get_total_people_desc
from diff_analysis_of_influencing_factors.data_set.columns_name import exgColumnsName
from matplotlib import pyplot
from xgboost import XGBClassifier
from xgboost import plot_importance
# from data_preprocess.columnsExg import *
import lightgbm as lgb
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from chinese_calendar import is_workday, is_holiday

# sns.set_style('white',{'font.sans-serif':['simhei','Arial']})
sns.set(style='whitegrid',palette='tab10')
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['font.family'] = 'SimHei'
matplotlib.rcParams['font.size'] = 10
matplotlib.rcParams['axes.unicode_minus']=False
plt.rcParams['font.sans-serif']=['Microsoft YaHei']

# 分析单商品在所有油站销量影响因素

# 首先获取所有相关数据
def do_analysis_number_of_people_all_plants( start_time, end_time):

    # 油站信息、poi信息，用于模型训练（已编码）
    data = get_total_people(start_time, end_time)
    # #对数转换
    # # log_col = ["number_station"]
    # # data = bill_log(data, log_col)
    # data['date'] = pd.DatetimeIndex(data['date']).date
    # data['year'] = pd.DatetimeIndex(data['date']).year
    # data['month'] = pd.DatetimeIndex(data['date']).month
    # data['weekday'] = pd.DatetimeIndex(data['date']).weekday + 1
    # data['is_holiday'] = data['date'].apply(lambda x: is_holiday(x))
    # data['is_holiday'] = data['is_holiday'].astype('int')

    # 更改列名
    data = exgColumnsName(data)

    # 划分数据集
    # 对销量进行分析
    target_name = '进站人数'


    # xgboost_s_time = time.time()
    # xgboost_mse, xgboost_rmse, xgboost_r2, xgboost_mae, xgboost_mape, xgboost_importance_dict = do_analysis_use_xgboost(data, target_name)
    # xgboost_e_time = time.time()
    # xgboost_time = xgboost_e_time - xgboost_s_time

    # lightGBM_s_time = time.time()
    # lightGBM_mse, lightGBM_rmse, lightGBM_r2, lightGBM_mae, lightGBM_mape, lightGBM_importance_dict = do_analysis_use_lightGBM(data, target_name)
    # lightGBM_e_time = time.time()
    # lightGBM_time = lightGBM_e_time - lightGBM_s_time

    # lasso_xgboost_s_time = time.time()
    # lasso_xgboost_mse, lasso_xgboost_rmse, lasso_xgboost_r2, lasso_xgboost_mae, lasso_xgboost_mape, lasso_xgboost_importance_dict = do_analysis_use_lasso_xgboost(data, target_name)
    # lasso_xgboost_e_time = time.time()
    # lasso_xgboost_time = lasso_xgboost_e_time - lasso_xgboost_s_time

    lasso_lightGBM_s_time = time.time()
    lasso_lightGBM_mse, lasso_lightGBM_rmse, lasso_lightGBM_r2, lasso_lightGBM_mae, lasso_lightGBM_mape, lasso_lightGBM_importance_dict = do_analysis_use_esf_gbdt(
        data, target_name)
    lasso_lightGBM_e_time = time.time()
    lasso_lightGBM_time = lasso_lightGBM_e_time - lasso_lightGBM_s_time
    #
    # print('[info : xgboost] The mse of prediction is:', xgboost_mse)
    # print('[info : xgboost] The rmse of prediction is:', xgboost_rmse)  # 计算真实值和预测值之间的均方根误差
    # print('[info : xgboost] the r2 of prediction is:', xgboost_r2)
    # print('[info : xgboost] the mae of prediction is:', xgboost_mae)
    # print('[info : xgboost] the mape of prediction is:', xgboost_mape)
    # #
    # print('[info : lightGBM] The mse of prediction is:', lightGBM_mse)
    # print('[info : lightGBM] The rmse of prediction is:', lightGBM_rmse)  # 计算真实值和预测值之间的均方根误差
    # print('[info : lightGBM] the r2 of prediction is:', lightGBM_r2)
    # print('[info : lightGBM] the mae of prediction is:', lightGBM_mae)
    # print('[info : lightGBM] the mape of prediction is:', lightGBM_mape)
    #
    # print('[info : lasso_xgboost] The mse of prediction is:', lasso_xgboost_mse)
    # print('[info : lasso_xgboost] The rmse of prediction is:', lasso_xgboost_rmse)  # 计算真实值和预测值之间的均方根误差
    # print('[info : lasso_xgboost] the r2 of prediction is:', lasso_xgboost_r2)
    # print('[info : lasso_xgboost] the mae of prediction is:', lasso_xgboost_mae)
    # print('[info : lasso_xgboost] the mape of prediction is:', lasso_xgboost_mape)

    print('[info : lasso_lightGBM] The mse of prediction is:', lasso_lightGBM_mse)
    print('[info : lasso_lightGBM] The rmse of prediction is:', lasso_lightGBM_rmse)  # 计算真实值和预测值之间的均方根误差
    print('[info : lasso_lightGBM] the r2 of prediction is:', lasso_lightGBM_r2)
    print('[info : lasso_lightGBM] the mae of prediction is:', lasso_lightGBM_mae)
    print('[info : lasso_lightGBM] the mape of prediction is:', lasso_lightGBM_mape)

  #  print( 'xgboost_time: ', xgboost_time)
  #  #  print('lightGBM_time: ', lightGBM_time)
  #   print('lasso_lightGBM_time: ', lasso_lightGBM_time)



    # print(importance_dict)
    # plot_importance(importance_dict)
    # plt.tick_params(labelsize=6)
    # plt.savefig('进站人数1.png', bbox_inches='tight')
    pyplot.show()

    # 获取该商品的销量描述信息、油站描述信息、poi信息等（未编码）展示上面得出的影响因素与销量的具体影响
    # data_desc = get_total_people_desc(start_time, end_time)
    #
    # data_desc['date'] = pd.DatetimeIndex(data_desc['date']).date
    # data_desc['year'] = pd.DatetimeIndex(data_desc['date']).year
    # data_desc['month'] = pd.DatetimeIndex(data_desc['date']).month
    # data_desc['weekday'] = pd.DatetimeIndex(data_desc['date']).weekday + 1
    # data_desc['is_holiday'] = data_desc['date'].apply(lambda x: is_holiday(x))
    # data_desc['is_holiday'] = data_desc['is_holiday'].astype('int')

    # # 更改列名
    # data_desc = exgColumnsName(data_desc)
    #
    #
    # g1 = sns.pairplot(data_desc,
    #              height=4,
    #              x_vars=['油站类别'],
    #              y_vars=['进站人数'], plot_kws={'alpha': 0.1})
    # g1.fig.set_size_inches(25, 5)
    # g1.savefig('进站人数油站类别.png', bbox_inches='tight')
    #
    # g2 = sns.pairplot(data_desc,
    #              height=4,
    #              x_vars=['加油站位置'],
    #              y_vars=['进站人数'], plot_kws={'alpha': 0.1})
    # g2.fig.set_size_inches(15, 5)
    # g2.savefig('进站人数油站位置.png', bbox_inches='tight')
    #
    # g3 = sns.pairplot(data_desc,
    #                   height=4,
    #                   x_vars=['道路等级'],
    #                   y_vars=['进站人数'], plot_kws={'alpha': 0.1})
    # #g3.fig.set_size_inches(30, 5)
    # g3.savefig('进站人数道路等级.png', bbox_inches='tight')
    #
    # g4 = sns.pairplot(data_desc,
    #                   height=4,
    #                   x_vars=['月份'],
    #                   y_vars=['进站人数'], plot_kws={'alpha': 0.1})
    # # g4.fig.set_size_inches(30, 5)
    # g4.savefig('香烟类油品月份.png', bbox_inches='tight')
    # g5 = sns.pairplot(data_desc,
    #                   height=4,
    #                   x_vars=['星期'],
    #                   y_vars=['进站人数'], plot_kws={'alpha': 0.1})
    # # g4.fig.set_size_inches(30, 5)
    # g5.savefig('进站人数星期.png', bbox_inches='tight')
    # g6 = sns.pairplot(data_desc,
    #                   height=4,
    #                   x_vars=['节假日'],
    #                   y_vars=['进站人数'], plot_kws={'alpha': 0.1})
    # # g4.fig.set_size_inches(30, 5)
    # g6.savefig('进站人数节假日.png', bbox_inches='tight')
    # g7 = sns.pairplot(data_desc,
    #                   height=4,
    #                   x_vars=['生活服务POI数量'],
    #                   y_vars=['进站人数'], plot_kws={'alpha': 0.1})
    # g7.fig.set_size_inches(25, 5)
    # g7.savefig('进站人数生活服务.png', bbox_inches='tight')
    # g8 = sns.pairplot(data_desc,
    #                   height=4,
    #                   x_vars=['购物服务POI数量'],
    #                   y_vars=['进站人数'], plot_kws={'alpha': 0.1})
    # g8.fig.set_size_inches(25, 5)
    # g8.savefig('进站人数购物服务.png', bbox_inches='tight')
    # g9 = sns.pairplot(data_desc,
    #                   height=4,
    #                   x_vars=['商务住宅POI数量'],
    #                   y_vars=['进站人数'], plot_kws={'alpha': 0.1})
    # g9.fig.set_size_inches(25, 5)
    # g9.savefig('进站人数油站类别.png', bbox_inches='tight')
    #
    # g10 = sns.pairplot(data_desc,
    #                   height=4,
    #                   x_vars=['公司企业POI数量'],
    #                   y_vars=['进站人数'], plot_kws={'alpha': 0.1})
    # g10.fig.set_size_inches(25, 5)
    # g10.savefig('进站人数公司企业.png', bbox_inches='tight')
    # g11 = sns.pairplot(data_desc,
    #                    height=4,
    #                    x_vars=['餐饮服务POI数量'],
    #                    y_vars=['进站人数'], plot_kws={'alpha': 0.1})
    # g11.fig.set_size_inches(25, 5)
    # g11.savefig('进站人数餐饮服务.png', bbox_inches='tight')
    #
    #
    # plt.show()

    # 通过上面的特征重要度dict 来进行字段筛选

    pass


do_analysis_number_of_people_all_plants('2019-01-01', '2019-12-31')