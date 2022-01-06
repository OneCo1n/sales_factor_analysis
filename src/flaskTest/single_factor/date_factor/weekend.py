from flaskTest.views.data_views.total_views import get_total_by_material

from chinese_calendar import is_holiday
import seaborn as sns
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

sns.set(style='whitegrid',palette='tab10')
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['font.family'] = 'SimHei'
matplotlib.rcParams['font.size'] = 10
matplotlib.rcParams['axes.unicode_minus']=False

data = get_total_by_material('000000000070251989')



data['date']=pd.DatetimeIndex(data['date']).date
data['year']=pd.DatetimeIndex(data['date']).year
data['month']=pd.DatetimeIndex(data['date']).month
data['weekday']=pd.DatetimeIndex(data['date']).weekday + 1
data['is_holiday'] = data['date'].apply(lambda x:is_holiday(x))
data['is_holiday'] = data['is_holiday'].astype('int')
data['month'] = data['month'].astype('int')
data['weekday'] = data['weekday'].astype('int')


print(data.dtypes)
print(data)
print(data['month'])
print(data['weekday'])
print(list(data['is_holiday']))

sales_date_list = []

for index, row in data.iterrows():
    sale_dict = {
        '月份': row['month'],
        '星期': row['weekday'],
        '节假日': row['is_holiday'],
        '销量': float(row['quantity'])
    }
    sales_date_list.append(sale_dict)

print(sales_date_list)


sns.pairplot(data ,
             height=4,
             x_vars=['month','weekday','is_holiday'] ,
             y_vars=['quantity'] , plot_kws={'alpha': 0.1})
plt.show()