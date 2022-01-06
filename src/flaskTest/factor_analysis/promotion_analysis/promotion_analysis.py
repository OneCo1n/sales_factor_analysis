from flaskTest.views.data_views.total_views import get_total_by_material, get_bill_and_plant_desc_by_material, \
    get_bill_and_promotion_by_material
import seaborn as sns
import matplotlib

import matplotlib.pyplot as plt
# sns.set_style('white',{'font.sans-serif':['simhei','Arial']})
sns.set(style='whitegrid',palette='tab10')
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['font.family'] = 'SimHei'
matplotlib.rcParams['font.size'] = 10
matplotlib.rcParams['axes.unicode_minus']=False

def sales_analysis_on_promotion(material):

    # data = get_bill_and_plant_desc_by_material(material)
    data = get_bill_and_promotion_by_material(material)



    sns.pairplot(data,
                 height=4,
                 x_vars=['promotion_type'],
                 y_vars=['quantity'], plot_kws={'alpha': 0.1})

    g2 = sns.pairplot(data,
                 height=4,
                 x_vars=['promotion_desc'],
                 y_vars=['quantity'], plot_kws={'alpha': 0.1})
    g2.fig.set_size_inches(10, 5)
    plt.show()

sales_analysis_on_promotion('000000000070251989')