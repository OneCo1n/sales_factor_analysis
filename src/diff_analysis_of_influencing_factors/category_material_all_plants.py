import time

from diff_analysis_of_influencing_factors.data_set.data_division import model_data_division
from diff_analysis_of_influencing_factors.train_model.AKL_PSO_lightGBM.AKL_PSO_lightGBM import \
    do_analysis_use_AKL_PSO_lightGBM
from diff_analysis_of_influencing_factors.train_model.AKL_lightGBM.AKL_lightGBM import do_analysis_use_AKL_lightGBM
from diff_analysis_of_influencing_factors.train_model.esgc.esgc import do_analysis_use_esgc
from diff_analysis_of_influencing_factors.train_model.gbrt.gbrt import do_analysis_use_gbrt
from diff_analysis_of_influencing_factors.train_model.lasso_lightGBM.esf_gbdt import \
    do_analysis_use_esf_gbdt
from diff_analysis_of_influencing_factors.train_model.lasso_lightGBM.lasso_lightGBM import \
    do_analysis_use_lasso_lightGBM
from diff_analysis_of_influencing_factors.train_model.lasso_xgboost.lasso_xgboost import do_analysis_use_lasso_xgboost
from diff_analysis_of_influencing_factors.train_model.lightGBM.lightGBM import use_lightGBM, do_analysis_use_lightGBM
from diff_analysis_of_influencing_factors.train_model.xgboost.xgboost import do_analysis_use_xgboost
from flaskTest.views.data_views.total_views import get_total_by_material, get_total_desc_by_material, \
    get_total_by_category, get_total_desc_by_category
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

# sns.set_style('white',{'font.sans-serif':['simhei','Arial']})
sns.set(style='whitegrid',palette='tab10')
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['font.family'] = 'SimHei'
matplotlib.rcParams['font.size'] = 10
matplotlib.rcParams['axes.unicode_minus']=False
plt.rcParams['font.sans-serif']=['Microsoft YaHei']


def do_analysis_category_material_all_plants(experiment_times, category, model_dict):
    start_time = '2019-01-01'
    end_time = '2019-12-31'
    # 获取该商品的销量数据、油站信息、poi信息、促销信息等，用于模型训练（已编码）
    data = get_total_by_category(category, start_time, end_time)

    # 更改列名
    data = exgColumnsName(data)

    # 划分数据集
    # 对销量进行分析
    data.loc[data['quantity'] == 0] = 1;
    target_name = 'quantity'

    # 实验结果文件
    filename_category_result = 'experiments_result_' + category + '.txt'


    # XGBoost
    if (model_dict['xgboost'] == 1):
        xgboost_mse_total = 0.0
        xgboost_rmse_total = 0.0
        # xgboost_r2_total = 0.0
        xgboost_mae_total = 0.0
        xgboost_mape_total = 0.0
        xgboost_time_total = 0.0
        for i in range(experiment_times):
            xgboost_s_time = time.time()
            xgboost_mse, xgboost_rmse, xgboost_r2, xgboost_mae, xgboost_mape, xgboost_importance_dict = do_analysis_use_xgboost(data, target_name)
            xgboost_e_time = time.time()
            xgboost_time = xgboost_e_time - xgboost_s_time
            xgboost_mse_total += xgboost_mse
            xgboost_rmse_total += xgboost_rmse
            # xgboost_r2_total += xgboost_r2
            xgboost_mae_total += xgboost_mae
            xgboost_mape_total += xgboost_mape
            xgboost_time_total += xgboost_time

            with open(filename_category_result, 'a') as file_object:
                file_object.write("xgboost number of times: " + str(i))
                file_object.write("\n")
                file_object.write("xgboost_mse:  " + str(xgboost_mse))
                file_object.write("\n")
                file_object.write("xgboost_rmse: " + str(xgboost_rmse))
                file_object.write("\n")
                file_object.write("xgboost_mae:  " + str(xgboost_mae))
                file_object.write("\n")
                file_object.write("xgboost_mape: " + str(xgboost_mape))
                file_object.write("\n")
                # file_object.write("xgboost_r2:   " + str(xgboost_r2))
                # file_object.write("\n")
                file_object.write("xgboost_time: " + str(xgboost_time) + " s")
                file_object.write("\n")
                if (i == experiment_times - 1):
                    file_object.write("------------------------------------------------------------")
                    file_object.write("\n")
                    file_object.write("xgboost average of " + str(experiment_times) + " results: ")
                    file_object.write("\n")
                    file_object.write("xgboost_mse:  " + str(xgboost_mse_total / experiment_times))
                    file_object.write("\n")
                    file_object.write("xgboost_rmse: " + str(xgboost_rmse_total / experiment_times))
                    file_object.write("\n")
                    file_object.write("xgboost_mae:  " + str(xgboost_mae_total / experiment_times))
                    file_object.write("\n")
                    file_object.write("xgboost_mape: " + str(xgboost_mape_total / experiment_times))
                    file_object.write("\n")
                    # file_object.write("xgboost_r2:   " + str(xgboost_r2_total / experiment_times))
                    # file_object.write("\n")
                    file_object.write("xgboost_time: " + str(xgboost_time_total / experiment_times) + " s")
                    file_object.write("\n")
                    file_object.write("------------------------------------------------------------")
                    file_object.write("\n")


    # lightGBM
    if (model_dict['lightGBM'] == 1):
        lightGBM_mse_total = 0.0
        lightGBM_rmse_total = 0.0
        # lightGBM_r2_total = 0.0
        lightGBM_mae_total = 0.0
        lightGBM_mape_total = 0.0
        lightGBM_time_total = 0.0
        for i in range(experiment_times):
            lightGBM_s_time = time.time()
            lightGBM_mse, lightGBM_rmse, lightGBM_r2, lightGBM_mae, lightGBM_mape, lightGBM_importance_dict = do_analysis_use_lightGBM(data, target_name)
            lightGBM_e_time = time.time()
            lightGBM_time = lightGBM_e_time - lightGBM_s_time

            lightGBM_mse_total += lightGBM_mse
            lightGBM_rmse_total += lightGBM_rmse
            # lightGBM_r2_total += lightGBM_r2
            lightGBM_mae_total += lightGBM_mae
            lightGBM_mape_total += lightGBM_mape
            lightGBM_time_total += lightGBM_time

            with open(filename_category_result, 'a') as file_object:
                file_object.write("lightGBM number of times: " + str(i))
                file_object.write("\n")
                file_object.write("lightGBM_mse:  " + str(lightGBM_mse))
                file_object.write("\n")
                file_object.write("lightGBM_rmse: " + str(lightGBM_rmse))
                file_object.write("\n")
                file_object.write("lightGBM_mae:  " + str(lightGBM_mae))
                file_object.write("\n")
                file_object.write("lightGBM_mape: " + str(lightGBM_mape))
                file_object.write("\n")
                # file_object.write("lightGBM_r2:  " + str(lightGBM_r2))
                # file_object.write("\n")
                file_object.write("lightGBM_time: " + str(lightGBM_time) + " s" )
                file_object.write("\n")
                if (i == experiment_times - 1):
                    file_object.write("------------------------------------------------------------")
                    file_object.write("\n")
                    file_object.write("lightGBM average of " + str(experiment_times) + " results: ")
                    file_object.write("\n")
                    file_object.write("lightGBM_mse:  " + str(lightGBM_mse_total/experiment_times))
                    file_object.write("\n")
                    file_object.write("lightGBM_rmse: " + str(lightGBM_rmse_total/experiment_times))
                    file_object.write("\n")
                    file_object.write("lightGBM_mae:  " + str(lightGBM_mae_total/experiment_times))
                    file_object.write("\n")
                    file_object.write("lightGBM_mape: " + str(lightGBM_mape_total/experiment_times))
                    file_object.write("\n")
                    # file_object.write("lightGBM_r2:  " + str(lightGBM_r2_total/experiment_times))
                    # file_object.write("\n")
                    file_object.write("lightGBM_time: " + str(lightGBM_time_total/experiment_times) + " s")
                    file_object.write("\n")
                    file_object.write("------------------------------------------------------------")
                    file_object.write("\n")

    # lasso_lightGBM
    if (model_dict['lasso_lightGBM'] == 1):
        lasso_lightGBM_mse_total = 0.0
        lasso_lightGBM_rmse_total = 0.0
        # lasso_lightGBM_r2_total = 0.0
        lasso_lightGBM_mae_total = 0.0
        lasso_lightGBM_mape_total = 0.0
        lasso_lightGBM_time_total = 0.0
        for i in range(experiment_times):
            lasso_lightGBM_s_time = time.time()
            lasso_lightGBM_mse, lasso_lightGBM_rmse, lasso_lightGBM_r2, lasso_lightGBM_mae, lasso_lightGBM_mape, lasso_lightGBM_importance_dict = do_analysis_use_lasso_lightGBM(data, target_name)
            lasso_lightGBM_e_time = time.time()
            lasso_lightGBM_time = lasso_lightGBM_e_time - lasso_lightGBM_s_time

            lasso_lightGBM_mse_total += lasso_lightGBM_mse
            lasso_lightGBM_rmse_total += lasso_lightGBM_rmse
            # lasso_lightGBM_r2_total += lasso_lightGBM_r2
            lasso_lightGBM_mae_total += lasso_lightGBM_mae
            lasso_lightGBM_mape_total += lasso_lightGBM_mape
            lasso_lightGBM_time_total += lasso_lightGBM_time

            with open(filename_category_result, 'a') as file_object:
                file_object.write("lasso_lightGBM number of times: " + str(i))
                file_object.write("\n")
                file_object.write("lasso_lightGBM_mse:  " + str(lasso_lightGBM_mse))
                file_object.write("\n")
                file_object.write("lasso_lightGBM_rmse: " + str(lasso_lightGBM_rmse))
                file_object.write("\n")
                file_object.write("lasso_lightGBM_mae:  " + str(lasso_lightGBM_mae))
                file_object.write("\n")
                file_object.write("lasso_lightGBM_mape: " + str(lasso_lightGBM_mape))
                file_object.write("\n")
                # file_object.write("lasso_lightGBM_r2:   " + str(lasso_lightGBM_r2))
                # file_object.write("\n")
                file_object.write("lasso_lightGBM_time: " + str(lasso_lightGBM_time) + " s" )
                file_object.write("\n")
                if (i == experiment_times - 1):
                    file_object.write("------------------------------------------------------------")
                    file_object.write("\n")
                    file_object.write("lasso_lightGBM average of " + str(experiment_times) + " results: ")
                    file_object.write("\n")
                    file_object.write("lasso_lightGBM_mse:  " + str(lasso_lightGBM_mse_total / experiment_times))
                    file_object.write("\n")
                    file_object.write("lasso_lightGBM_rmse: " + str(lasso_lightGBM_rmse_total / experiment_times))
                    file_object.write("\n")
                    file_object.write("lasso_lightGBM_mae:  " + str(lasso_lightGBM_mae_total / experiment_times))
                    file_object.write("\n")
                    file_object.write("lasso_lightGBM_mape: " + str(lasso_lightGBM_mape_total / experiment_times))
                    file_object.write("\n")
                    # file_object.write("lasso_lightGBM_mse:  " + str(lasso_lightGBM_r2_total / experiment_times))
                    # file_object.write("\n")
                    file_object.write("lasso_lightGBM_time: " + str(lasso_lightGBM_time_total / experiment_times) + " s")
                    file_object.write("\n")
                    file_object.write("------------------------------------------------------------")
                    file_object.write("\n")



############################
    # AKL_lightGBM
    if (model_dict['AKL_lightGBM'] == 1):
        AKL_lightGBM_mse_total = 0.0
        AKL_lightGBM_rmse_total = 0.0
        # AKL_lightGBM_r2_total = 0.0
        AKL_lightGBM_mae_total = 0.0
        AKL_lightGBM_mape_total = 0.0
        AKL_lightGBM_time_total = 0.0
        for i in range(experiment_times):
            AKL_lightGBM_s_time = time.time()
            AKL_lightGBM_mse, AKL_lightGBM_rmse, AKL_lightGBM_r2, AKL_lightGBM_mae, AKL_lightGBM_mape, AKL_lightGBM_importance_dict = do_analysis_use_AKL_lightGBM(
                    data, target_name)
            AKL_lightGBM_e_time = time.time()
            AKL_lightGBM_time = AKL_lightGBM_e_time - AKL_lightGBM_s_time

            AKL_lightGBM_mse_total += AKL_lightGBM_mse
            AKL_lightGBM_rmse_total += AKL_lightGBM_rmse
            # AKL_lightGBM_r2_total += AKL_lightGBM_r2
            AKL_lightGBM_mae_total += AKL_lightGBM_mae
            AKL_lightGBM_mape_total += AKL_lightGBM_mape
            AKL_lightGBM_time_total += AKL_lightGBM_time

            with open(filename_category_result, 'a') as file_object:
                file_object.write("AKL_lightGBM number of times: " + str(i))
                file_object.write("\n")
                file_object.write("AKL_lightGBM_mse:  " + str(AKL_lightGBM_mse))
                file_object.write("\n")
                file_object.write("AKL_lightGBM_rmse: " + str(AKL_lightGBM_rmse))
                file_object.write("\n")
                file_object.write("AKL_lightGBM_mae:  " + str(AKL_lightGBM_mae))
                file_object.write("\n")
                file_object.write("AKL_lightGBM_mape: " + str(AKL_lightGBM_mape))
                file_object.write("\n")
                # file_object.write("AKL_lightGBM_r2:   " + str(AKL_lightGBM_r2))
                # file_object.write("\n")
                file_object.write("AKL_lightGBM_time: " + str(AKL_lightGBM_time) + " s")
                file_object.write("\n")
                if (i == experiment_times - 1):
                    file_object.write("------------------------------------------------------------")
                    file_object.write("\n")
                    file_object.write("AKL_lightGBM average of " + str(experiment_times) + " results: ")
                    file_object.write("\n")
                    file_object.write("AKL_lightGBM_mse:  " + str(AKL_lightGBM_mse_total / experiment_times))
                    file_object.write("\n")
                    file_object.write("AKL_lightGBM_rmse: " + str(AKL_lightGBM_rmse_total / experiment_times))
                    file_object.write("\n")
                    file_object.write("AKL_lightGBM_mae:  " + str(AKL_lightGBM_mae_total / experiment_times))
                    file_object.write("\n")
                    file_object.write("AKL_lightGBM_mape: " + str(AKL_lightGBM_mape_total / experiment_times))
                    file_object.write("\n")
                    # file_object.write("AKL_lightGBM_r2:  " + str(AKL_lightGBM_r2_total / experiment_times))
                    # file_object.write("\n")
                    file_object.write(
                            "AKL_lightGBM_time: " + str(AKL_lightGBM_time_total / experiment_times) + " s")
                    file_object.write("\n")
                    file_object.write("------------------------------------------------------------")
                    file_object.write("\n")

    #AKL_PSO_lightGBM
    if (model_dict['AKL_PSO_lightGBM'] == 1):
        AKL_PSO_lightGBM_mse_total = 0.0
        AKL_PSO_lightGBM_rmse_total = 0.0
        # AKL_PSO_lightGBM_r2_total = 0.0
        AKL_PSO_lightGBM_mae_total = 0.0
        AKL_PSO_lightGBM_mape_total = 0.0
        AKL_PSO_lightGBM_time_total = 0.0
        for i in range(experiment_times):
            AKL_PSO_lightGBM_s_time = time.time()
            AKL_PSO_lightGBM_mse, AKL_PSO_lightGBM_rmse, AKL_PSO_lightGBM_r2, AKL_PSO_lightGBM_mae, AKL_PSO_lightGBM_mape, AKL_PSO_lightGBM_importance_dict = do_analysis_use_AKL_PSO_lightGBM(
                data, target_name)
            AKL_PSO_lightGBM_e_time = time.time()
            AKL_PSO_lightGBM_time = AKL_PSO_lightGBM_e_time - AKL_PSO_lightGBM_s_time

            AKL_PSO_lightGBM_mse_total += AKL_PSO_lightGBM_mse
            AKL_PSO_lightGBM_rmse_total += AKL_PSO_lightGBM_rmse
            # AKL_PSO_lightGBM_r2_total += AKL_PSO_lightGBM_r2
            AKL_PSO_lightGBM_mae_total += AKL_PSO_lightGBM_mae
            AKL_PSO_lightGBM_mape_total += AKL_PSO_lightGBM_mape
            AKL_PSO_lightGBM_time_total += AKL_PSO_lightGBM_time

            with open(filename_category_result, 'a') as file_object:
                file_object.write("AKL_PSO_lightGBM number of times: " + str(i))
                file_object.write("\n")
                file_object.write("AKL_PSO_lightGBM_mse:  " + str(AKL_PSO_lightGBM_mse))
                file_object.write("\n")
                file_object.write("AKL_PSO_lightGBM_rmse: " + str(AKL_PSO_lightGBM_rmse))
                file_object.write("\n")
                file_object.write("AKL_PSO_lightGBM_mae:  " + str(AKL_PSO_lightGBM_mae))
                file_object.write("\n")
                file_object.write("AKL_PSO_lightGBM_mape: " + str(AKL_PSO_lightGBM_mape))
                file_object.write("\n")
                # file_object.write("AKL_PSO_lightGBM_r2:   " + str(AKL_PSO_lightGBM_r2))
                # file_object.write("\n")
                file_object.write("AKL_PSO_lightGBM_time: " + str(AKL_PSO_lightGBM_time) + " s")
                file_object.write("\n")
                if (i == experiment_times - 1):
                    file_object.write("------------------------------------------------------------")
                    file_object.write("\n")
                    file_object.write("AKL_PSO_lightGBM average of " + str(experiment_times) + " results: ")
                    file_object.write("\n")
                    file_object.write("AKL_PSO_lightGBM_mse:  " + str(AKL_PSO_lightGBM_mse_total / experiment_times))
                    file_object.write("\n")
                    file_object.write("AKL_PSO_lightGBM_rmse: " + str(AKL_PSO_lightGBM_rmse_total / experiment_times))
                    file_object.write("\n")
                    file_object.write("AKL_PSO_lightGBM_mae:  " + str(AKL_PSO_lightGBM_mae_total / experiment_times))
                    file_object.write("\n")
                    file_object.write("AKL_PSO_lightGBM_mape: " + str(AKL_PSO_lightGBM_mape_total / experiment_times))
                    file_object.write("\n")
                    # file_object.write("AKL_PSO_lightGBM_r2:  " + str(AKL_PSO_lightGBM_r2_total / experiment_times))
                    # file_object.write("\n")
                    file_object.write(
                        "AKL_PSO_lightGBM_time: " + str(AKL_PSO_lightGBM_time_total / experiment_times) + " s")
                    file_object.write("\n")
                    file_object.write("------------------------------------------------------------")
                    file_object.write("\n")


    # esgc
    if (model_dict['esgc'] == 1):
        esgc_mse_total = 0.0
        esgc_rmse_total = 0.0
        # esgc_r2_total = 0.0
        esgc_mae_total = 0.0
        esgc_mape_total = 0.0
        esgc_time_total = 0.0
        for i in range(experiment_times):
            esgc_s_time = time.time()
            esgc_mse, esgc_rmse, esgc_r2, esgc_mae, esgc_mape, esgc_importance_dict = do_analysis_use_esgc(data, target_name)
            esgc_e_time = time.time()
            esgc_time = esgc_e_time - esgc_s_time

            esgc_mse_total += esgc_mse
            esgc_rmse_total += esgc_rmse
            # esgc_r2_total += esgc_r2
            esgc_mae_total += esgc_mae
            esgc_mape_total += esgc_mape
            esgc_time_total += esgc_time

            with open(filename_category_result, 'a') as file_object:
                file_object.write("esgc number of times: " + str(i))
                file_object.write("\n")
                file_object.write("esgc_mse:  " + str(esgc_mse))
                file_object.write("\n")
                file_object.write("esgc_rmse: " + str(esgc_rmse))
                file_object.write("\n")
                file_object.write("esgc_mae:  " + str(esgc_mae))
                file_object.write("\n")
                file_object.write("esgc_mape: " + str(esgc_mape))
                file_object.write("\n")
                # file_object.write("esgc_r2:   " + str(esgc_r2))
                # file_object.write("\n")
                file_object.write("esgc_time: " + str(esgc_time) + " s" )
                file_object.write("\n")
                if (i == experiment_times - 1):
                    file_object.write("------------------------------------------------------------")
                    file_object.write("\n")
                    file_object.write("esgc average of " + str(experiment_times) + " results: ")
                    file_object.write("\n")
                    file_object.write("esgc_mse:  " + str(esgc_mse_total / experiment_times))
                    file_object.write("\n")
                    file_object.write("esgc_rmse: " + str(esgc_rmse_total / experiment_times))
                    file_object.write("\n")
                    file_object.write("esgc_mae:  " + str(esgc_mae_total / experiment_times))
                    file_object.write("\n")
                    file_object.write("esgc_mape: " + str(esgc_mape_total / experiment_times))
                    file_object.write("\n")
                    # file_object.write("esgc_r2:  " + str(esgc_r2_total / experiment_times))
                    # file_object.write("\n")
                    file_object.write("esgc_time: " + str(esgc_time_total / experiment_times) + " s")
                    file_object.write("\n")
                    file_object.write("------------------------------------------------------------")
                    file_object.write("\n")



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
    #
    # print('[info : lasso_lightGBM] The mse of prediction is:', lasso_lightGBM_mse)
    # print('[info : lasso_lightGBM] The rmse of prediction is:', lasso_lightGBM_rmse)  # 计算真实值和预测值之间的均方根误差
    # print('[info : lasso_lightGBM] the r2 of prediction is:', lasso_lightGBM_r2)
    # print('[info : lasso_lightGBM] the mae of prediction is:', lasso_lightGBM_mae)
    # print('[info : lasso_lightGBM] the mape of prediction is:', lasso_lightGBM_mape)

    # print('[info : esgc] The mse of prediction is:', esgc_mse)
    # print('[info : esgc] The rmse of prediction is:', esgc_rmse)  # 计算真实值和预测值之间的均方根误差
    # print('[info : esgc] the r2 of prediction is:', esgc_r2)
    # print('[info : esgc] the mae of prediction is:', esgc_mae)
    # print('[info : esgc] the mape of prediction is:', esgc_mape)


    # # 获取该商品的销量描述信息、油站描述信息、poi信息等（未编码）展示上面得出的影响因素与销量的具体影响
    # data_desc = get_total_desc_by_category(category, start_time, end_time)
    #
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
    # g1 = sns.pairplot(data_desc,
    #              height=4,
    #              x_vars=['油站类别'],
    #              y_vars=['quantity'], plot_kws={'alpha': 0.1})
    # g1.fig.set_size_inches(25, 5)
    # g1.savefig('香烟类油品油站类别.png', bbox_inches='tight')
    #
    # g2 = sns.pairplot(data_desc,
    #              height=4,
    #              x_vars=['加油站位置'],
    #              y_vars=['quantity'], plot_kws={'alpha': 0.1})
    # g2.fig.set_size_inches(15, 5)
    # g2.savefig('香烟类油品油站位置.png', bbox_inches='tight')
    #
    # g3 = sns.pairplot(data_desc,
    #                   height=4,
    #                   x_vars=['道路等级'],
    #                   y_vars=['quantity'], plot_kws={'alpha': 0.1})
    # #g3.fig.set_size_inches(30, 5)
    # g3.savefig('香烟类油品道路等级.png', bbox_inches='tight')
    #
    # g4 = sns.pairplot(data_desc,
    #                   height=4,
    #                   x_vars=['月份'],
    #                   y_vars=['quantity'], plot_kws={'alpha': 0.1})
    # # g4.fig.set_size_inches(30, 5)
    # g4.savefig('香烟类油品月份.png', bbox_inches='tight')
    # g5 = sns.pairplot(data_desc,
    #                   height=4,
    #                   x_vars=['星期'],
    #                   y_vars=['quantity'], plot_kws={'alpha': 0.1})
    # # g4.fig.set_size_inches(30, 5)
    # g5.savefig('香烟类油品星期.png', bbox_inches='tight')
    # g6 = sns.pairplot(data_desc,
    #                   height=4,
    #                   x_vars=['节假日'],
    #                   y_vars=['quantity'], plot_kws={'alpha': 0.1})
    # # g4.fig.set_size_inches(30, 5)
    # g6.savefig('香烟类油品节假日.png', bbox_inches='tight')
    #
    # plt.show()


    pass
#
# 润滑油
# 汽车用品
# 汽车服务
# 收费服务
# 包装饮料
# 个人护理用品
# 酒类
# 速食
# 日用品
# 饼干/糕点
# 糖果
# 奶类
# 清洁用品
# 零食
# 雪糕
# 面包
# 药品/计生/保健
# 家庭食品
# 香烟
# 办公图书音像
# 通讯/数码/电脑
# 其他
# 散装饮料
# 化工农资

# #
# do_analysis_category_material_all_plants('香烟','香烟', '2019-09-01', '2019-12-31')
# #
# do_analysis_category_material_all_plants('糖果','糖果', '2019-09-01', '2019-12-31')
#
# do_analysis_category_material_all_plants('糕点','饼干/糕点', '2019-09-01', '2019-12-31')
#
# do_analysis_category_material_all_plants('汽车用品', '汽车用品', '2019-09-01', '2019-12-31')
#
# do_analysis_category_material_all_plants('包装饮料','包装饮料', '2019-09-01', '2019-12-31')
