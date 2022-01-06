from flaskTest.views.data_views.total_views import get_total_by_material, get_bill_and_plant_desc_by_material, \
    get_bill_and_plant_desc_by_material_and_company
import seaborn as sns
import matplotlib

import matplotlib.pyplot as plt
# sns.set_style('white',{'font.sans-serif':['simhei','Arial']})
sns.set(style='whitegrid',palette='tab10')
sns.set(rc={'figure.figsize':(100,50)})
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['font.family'] = 'SimHei'
matplotlib.rcParams['font.size'] = 10
matplotlib.rcParams['axes.unicode_minus']=False

def sales_analysis_on_plants(material):

    data = get_bill_and_plant_desc_by_material(material)

    plt.figure(figsize=(44,8))
    sns.pairplot(data,
                 height=4,
                 x_vars=['plant_asset', 'road_class', 'plant_stars', 'store_class'],
                 y_vars=['quantity'], plot_kws={'alpha': 0.1})
    g = sns.pairplot(data,
                 height=4,
                 x_vars=['plant_class_code'],
                 y_vars=['quantity'], plot_kws={'alpha': 0.1})
    g.fig.set_size_inches(30, 5)
    g1 = sns.pairplot(data,
                 height=4,
                 x_vars=['plant_location_class'],
                 y_vars=['quantity'], plot_kws={'alpha': 0.1})
    g1.fig.set_size_inches(20, 5)
    g2 = sns.pairplot(data,
                 height=4,
                 x_vars=['plant_keyanliang_desc'],
                 y_vars=['quantity'], plot_kws={'alpha': 0.1})
    g2.fig.set_size_inches(10, 5)
    sns.pairplot(data,
                 height=4,
                 x_vars=['plant_type_desc'],
                 y_vars=['quantity'], plot_kws={'alpha': 0.1})
    plt.show()

def sales_analysis_on_company(material, company):

    data = get_bill_and_plant_desc_by_material_and_company(material, company)

    sns.pairplot(data,
                 height=4,
                 x_vars=['plant_asset', 'road_class', 'plant_stars', 'store_class'],
                 y_vars=['quantity'], plot_kws={'alpha': 0.1})
    g = sns.pairplot(data,
                     height=4,
                     x_vars=['plant_class_code'],
                     y_vars=['quantity'], plot_kws={'alpha': 0.1})
    g.fig.set_size_inches(30, 5)
    g1 = sns.pairplot(data,
                      height=4,
                      x_vars=['plant_location_class'],
                      y_vars=['quantity'], plot_kws={'alpha': 0.1})
    g1.fig.set_size_inches(20, 5)
    g2 = sns.pairplot(data,
                      height=4,
                      x_vars=['plant_keyanliang_desc'],
                      y_vars=['quantity'], plot_kws={'alpha': 0.1})
    g2.fig.set_size_inches(10, 5)
    sns.pairplot(data,
                 height=4,
                 x_vars=['plant_type_desc'],
                 y_vars=['quantity'], plot_kws={'alpha': 0.1})
    plt.show()


# sales_analysis_on_plants('000000000070251989')
# sales_analysis_on_company('000000000070251989', '北京销售北京分公司')
# sales_analysis_on_company('000000000070251989', '北京销售公司一分公司')
# sales_analysis_on_company('000000000070251989', '北京销售公司二分公司')
# sales_analysis_on_company('000000000070251989', '北京销售公司三分公司')
sales_analysis_on_company('000000000070251989', '北京销售公司四分公司')


