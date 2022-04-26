from diff_analysis_of_influencing_factors.data_set.data_division import model_data_division
from diff_analysis_of_influencing_factors.train_model.lasso_lightGBM.esf_gbdt import \
    do_analysis_use_esf_gbdt
from diff_analysis_of_influencing_factors.train_model.lasso_xgboost.lasso_xgboost import do_analysis_use_lasso_xgboost
from diff_analysis_of_influencing_factors.train_model.lightGBM.lightGBM import use_lightGBM, do_analysis_use_lightGBM
from diff_analysis_of_influencing_factors.train_model.xgboost.xgboost import do_analysis_use_xgboost
from flaskTest.views.data_views.total_views import get_total_by_material, get_total_desc_by_material, \
    get_total_by_material_and_company, get_total_desc_by_material_and_company, get_total_by_material_and_plant_class, \
    get_total_desc_by_material_and_plant_class, get_total_by_all_material_and_plant_class
from diff_analysis_of_influencing_factors.data_set.columns_name import exgColumnsName
from matplotlib import pyplot
from xgboost import XGBClassifier
from xgboost import plot_importance
from data_preprocess.columnsExg import *
import lightgbm as lgb
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
# sns.set_style('white',{'font.sans-serif':['simhei','Arial']})
sns.set(style='whitegrid',palette='tab10')
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['font.family'] = 'SimHei'
matplotlib.rcParams['font.size'] = 10
matplotlib.rcParams['axes.unicode_minus']=False

plt.rcParams['font.sans-serif']=['Microsoft YaHei']

# 分析单商品在所有油站销量影响因素

# 首先获取所有相关数据

def do_analysis_all_material_plant_group_by_plant_class(start_time, end_time, plant_class):


    # 获取该商品的销量数据、油站信息、poi信息、促销信息等，用于模型训练（已编码）
    data = get_total_by_all_material_and_plant_class(start_time, end_time, plant_class)

    # 更改列名
    data = exgColumnsName(data)

    # 划分数据集
    # 对销量进行分析
    target_name = 'quantity'

    x_mse, x_rmse, x_r2, x_mae, x_mape, x_importance_dict = do_analysis_use_xgboost(data, target_name)
    l_mse, l_rmse, l_r2, l_mae, l_mape, l_importance_dict = do_analysis_use_lightGBM(data, target_name)
    xl_mse, xl_rmse, xl_r2, xl_mae, xl_mape, xl_importance_dict = do_analysis_use_lasso_xgboost(data, target_name)
    ll_mse, ll_rmse, ll_r2, ll_mae, ll_mape, ll_importance_dict = do_analysis_use_esf_gbdt(data, target_name)


    # print(importance_dict)

    filename = 'shiyanjieguo.txt'
    with open(filename, 'a') as file_object:
        file_object.write(plant_class)
        file_object.write("\n")
        file_object.write("x_mse, x_rmse, x_r2, x_mae, x_mape\n")
        file_object.write(str(x_mse) + "  " + str(x_rmse) + "  " + str(x_r2) + "  " + str(x_mae) + "  " + str(x_mape))
        file_object.write("\n")
        file_object.write("l_mse, l_rmse, l_r2, l_mae, l_mape\n")
        file_object.write(str(l_mse) + "  " + str(l_rmse) + "  " + str(l_r2) + "  " + str(l_mae) + "  " + str(l_mape))
        file_object.write("\n")
        file_object.write("xl_mse, xl_rmse, xl_r2, xl_mae, xl_mape\n")
        file_object.write(str(xl_mse) + "  " + str(xl_rmse) + "  " + str(xl_r2) + "  " + str(xl_mae) + "  " + str(xl_mape))
        file_object.write("\n")
        file_object.write("ll_mse, ll_rmse, ll_r2, ll_mae, ll_mape\n")
        file_object.write(str(ll_mse) + "  " + str(ll_rmse) + "  " + str(ll_r2) + "  " + str(ll_mae) + "  " + str(ll_mape))
        file_object.write("\n")
        file_object.write("---------------\n")


    # plot_importance(importance_dict)
    # pyplot.show()

    # 获取该商品的销量描述信息、油站描述信息、poi信息等（未编码）展示上面得出的影响因素与销量的具体影响
    # data_desc = get_total_desc_by_material_and_plant_class(material, start_time, end_time, plant_class)
    # 更改列名
    # data_desc = exgColumnsName(data_desc)

    pass



# 0001	一类站 省会(直辖市)市区、地级市市
# 0002	二类站 高速公路、地级以上环城快速路
# 0003	三类站 县级市市区、县城城区、国道、
# 0004	四类站 县乡道路、乡镇、水上、农村

do_analysis_all_material_plant_group_by_plant_class('2019-09-01', '2019-12-01', '0001')
do_analysis_all_material_plant_group_by_plant_class('2019-09-01', '2019-12-01', '0002')
do_analysis_all_material_plant_group_by_plant_class('2019-09-01', '2019-12-01', '0003')
do_analysis_all_material_plant_group_by_plant_class('2019-09-01', '2019-12-01', '0004')