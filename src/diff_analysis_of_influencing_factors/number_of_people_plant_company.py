from diff_analysis_of_influencing_factors.data_set.data_division import model_data_division
from diff_analysis_of_influencing_factors.train_model.lasso_lightGBM.esf_gbdt import \
    do_analysis_use_esf_gbdt
from diff_analysis_of_influencing_factors.train_model.lightGBM.lightGBM import use_lightGBM, do_analysis_use_lightGBM
from flaskTest.views.data_views.total_views import get_total_by_material, get_total_desc_by_material, get_total_people, \
    get_total_people_desc, get_total_people_by_company, get_total_people_desc_by_company
from diff_analysis_of_influencing_factors.data_set.columns_name import exgColumnsName
from matplotlib import pyplot
from xgboost import XGBClassifier
from xgboost import plot_importance
#from data_preprocess.columnsExg import *
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
import pandas as pd
from chinese_calendar import is_workday, is_holiday

# 分析单商品在所有油站销量影响因素

# 首先获取所有相关数据
def do_analysis_number_of_people_plant_company(start_time, end_time, company):

    # 油站信息、poi信息，用于模型训练（已编码）
    data = get_total_people_by_company(start_time, end_time, company)
    data['plant'] = data['plant'].astype('float')

    data['date'] = pd.DatetimeIndex(data['date']).date
    data['year'] = pd.DatetimeIndex(data['date']).year
    data['month'] = pd.DatetimeIndex(data['date']).month
    data['weekday'] = pd.DatetimeIndex(data['date']).weekday + 1
    data['is_holiday'] = data['date'].apply(lambda x: is_holiday(x))
    data['is_holiday'] = data['is_holiday'].astype('int')

    # 更改列名
    print(data.dtypes)
    data = exgColumnsName(data)
    # 划分数据集
    # 对销量进行分析
    target_name = '进站人数'

    mse, rmse, r2, mae, mape, importance_dict = do_analysis_use_lightGBM(data, target_name)
    # mse, rmse, r2, importance_dict = do_analysis_use_lasso_xgboost(data, target_name)
    # mse, rmse, r2, mae, mape, importance_dict = do_analysis_use_esf_gbdt(data, target_name)

    print(importance_dict)
    plot_importance(importance_dict)
    plt.tick_params(labelsize=6)
    plt.savefig('进店人数.png', bbox_inches='tight')

    # 获取该商品的销量描述信息、油站描述信息、poi信息等（未编码）展示上面得出的影响因素与销量的具体影响
    data_desc = get_total_people_desc_by_company(start_time, end_time, company)
    # 更改列名
    data_desc = exgColumnsName(data_desc)

    plt.show()
    # 通过上面的特征重要度dict 来进行字段筛选

    pass
# '北京销售公司一分公司'
# '北京销售公司二分公司'
# '北京销售公司三分公司'
# '北京销售公司四分公司'
do_analysis_number_of_people_plant_company('2019-01-01', '2019-12-31', '北京销售公司一分公司')