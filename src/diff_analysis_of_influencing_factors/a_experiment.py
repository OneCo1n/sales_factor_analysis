from diff_analysis_of_influencing_factors.category_material_all_plants import do_analysis_category_material_all_plants
import time
import datetime



# 实验
# 对类别非油品进行不同模型的实验对比

# experiment_times  实验次数
# category_list   非油品类别列表
# 实验结果输出在 experiments_result_{category}.txt 文件中
# experiments_log 实验日志
# model_dict 参与实验的模型 1：进行实验 0：不进行

# 实验参数设置
experiment_times = 1
category_list = ["零食"]
experiments_log = 'a_experiments_log.txt'
model_dict = {'xgboost': 1, 'lightGBM': 0, 'lasso_lightGBM': 0, 'AKL_lightGBM': 0, 'AKL_PSO_lightGBM': 0, 'esgc': 0}


def do_experiments(experiment_times, category_list, experiments_log, model_dict):

    for category in category_list:
        start_time = time.time()
        with open(experiments_log, 'a') as file_object:
            file_object.write("experiments on " + category + " start at : " + str(datetime.datetime.now()))
            file_object.write("\n")
        do_analysis_category_material_all_plants(experiment_times, category, model_dict)
        end_time = time.time()
        experiments_time = end_time - start_time
        with open(experiments_log, 'a') as file_object:
            file_object.write("experiments on " + category + " end at : " + str(datetime.datetime.now()))
            file_object.write("\n")
            file_object.write("duration is : " + str(experiments_time))
            file_object.write("\n")


do_experiments(experiment_times, category_list, experiments_log, model_dict)


