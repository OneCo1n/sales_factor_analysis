import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


#训练集和测试集
train = pd.read_csv('../data/sales_train.csv')
test= pd.read_csv('../data/test.csv')
train.head()

test.head()

print('训练集的商店数量： %d ，商品数量： %d；\n' % (train['shop_id'].unique().size, train['item_id'].unique().size),
     '测试的商店数量： %d，商品数量： %d。' % (test['shop_id'].unique().size, test['item_id'].unique().size))

test[~test['shop_id'].isin(train['shop_id'].unique())]

test[~test['item_id'].isin(train['item_id'].unique())]['item_id'].unique()[:10]

np.array([5320, 5268, 5826, 3538, 3571, 3604, 3407, 3408, 3405, 3984], dtype=np.int64)

#商店数据集
shops = pd.read_csv('../data/shops.csv')
shops.head()

# 查看测试集是否包含了这几个商店
test[test['shop_id'].isin([39, 40, 10, 11, 0, 57, 58, 1, 12 ,56])]['shop_id'].unique()

np.array([10, 12, 57, 58, 56, 39], dtype=np.int64)

shop_id_map = {11: 10, 0: 57, 1: 58, 40: 39}
train.loc[train['shop_id'].isin(shop_id_map), 'shop_id'] = train.loc[train['shop_id'].isin(shop_id_map), 'shop_id'].map(shop_id_map)
train.loc[train['shop_id'].isin(shop_id_map), 'shop_id']

#Series([], Name: shop_id, dtype: int64)
pd.Series(['Name', 'dtype'],index= ['shop_id', 'int64'])
train.loc[train['shop_id'].isin([39, 40, 10, 11, 0, 57, 58, 1]), 'shop_id'].unique()
np.array([57, 58, 10, 39], dtype=np.int64)

shops['shop_city'] = shops['shop_name'].map(lambda x:x.split(' ')[0].strip('!'))
shop_types = ['ТЦ', 'ТРК', 'ТРЦ', 'ТК', 'МТРЦ']
shops['shop_type'] = shops['shop_name'].map(lambda x:x.split(' ')[1] if x.split(' ')[1] in shop_types else 'Others')
shops.loc[shops['shop_id'].isin([12, 56]), ['shop_city', 'shop_type']] = 'Online'  # 12和56号是网上商店
shops.head(13)

# 对商店信息进行编码，降低模型训练的内存消耗
shop_city_map = dict([(v,k) for k, v in enumerate(shops['shop_city'].unique())])
shop_type_map = dict([(v,k) for k, v in enumerate(shops['shop_type'].unique())])
shops['shop_city_code'] = shops['shop_city'].map(shop_city_map)
shops['shop_type_code'] = shops['shop_type'].map(shop_type_map)
shops.head(7)

items = pd.read_csv('../data/items.csv')
items

# 数据集比较大，只分析有没有重复名称不同ID的商品
items['item_name'] = items['item_name'].map(lambda x: ''.join(x.split(' ')))  # 删除空格
duplicated_item_name = items[items['item_name'].duplicated()]
duplicated_item_name

duplicated_item_name_rec = items[items['item_name'].isin(duplicated_item_name['item_name'])]  # 6个商品相同名字不同id的记录
duplicated_item_name_rec

test[test['item_id'].isin(duplicated_item_name_rec['item_id'])]['item_id'].unique()

np.array([19581, 5063], dtype=np.int64)

old_id = duplicated_item_name_rec['item_id'].values[::2]
new_id = duplicated_item_name_rec['item_id'].values[1::2]
old_new_map = dict(zip(old_id, new_id))
old_new_map

{2514: 2558, 2968: 2970, 5061: 5063, 14537: 14539, 19465: 19475, 19579: 19581}
train.loc[train['item_id'].isin(old_id), 'item_id'] = train.loc[train['item_id'].isin(old_id), 'item_id'].map(old_new_map)
train[train['item_id'].isin(old_id)]

train[train['item_id'].isin(duplicated_item_name_rec['item_id'].values)]['item_id'].unique()  # 旧id成功替换成新id
np.array([ 2558, 14539, 19475, 19581,  5063,  2970], dtype=np.int64)

items.groupby('item_id').size()[items.groupby('item_id').size() > 1]  # 检查同一个商品是否分了不同类目

#Series([], dtype: int64)
pd.Series(['dtype'], index= ['int64'])
cat = pd.read_csv('../data/item_categories.csv')
cat

cat[cat['item_category_name'].duplicated()]
cat['item_type'] = cat['item_category_name'].map(lambda x: 'Игры' if x.find('Игры ')>0 else x.split(' -')[0].strip('\"'))
cat.iloc[[32, 33, 34, -3, -2, -1]]  # 有几个比较特殊，需要另外调整一下
cat.iloc[[32,-3, -2], -1] = ['Карты оплаты', 'Чистые носители', 'Чистые носители' ]
cat.iloc[[32,-3, -2]]
item_type_map = dict([(v,k) for k, v in enumerate(cat['item_type'].unique())])
cat['item_type_code'] = cat['item_type'].map(item_type_map)
cat.head()
cat['sub_type'] = cat['item_category_name'].map(lambda x: x.split('-',1)[-1])
cat
cat['sub_type'].unique()
np.array([' Гарнитуры/Наушники', ' PS2', ' PS3', ' PS4', ' PSP', ' PSVita',
       ' XBOX 360', ' XBOX ONE', 'Билеты (Цифра)', 'Доставка товара',
       ' Прочие', ' Аксессуары для игр', ' Цифра',
       ' Дополнительные издания', ' Коллекционные издания',
       ' Стандартные издания', 'Карты оплаты (Кино, Музыка, Игры)',
       ' Live!', ' Live! (Цифра)', ' PSN', ' Windows (Цифра)', ' Blu-Ray',
       ' Blu-Ray 3D', ' Blu-Ray 4K', ' DVD', ' Коллекционное',
       ' Артбуки, энциклопедии', ' Аудиокниги', ' Аудиокниги (Цифра)',
       ' Аудиокниги 1С', ' Бизнес литература', ' Комиксы, манга',
       ' Компьютерная литература', ' Методические материалы 1С',
       ' Открытки', ' Познавательная литература', ' Путеводители',
       ' Художественная литература', ' CD локального производства',
       ' CD фирменного производства', ' MP3', ' Винил',
       ' Музыкальное видео', ' Подарочные издания', ' Атрибутика',
       ' Гаджеты, роботы, спорт', ' Мягкие игрушки', ' Настольные игры',
       ' Настольные игры (компактные)', ' Открытки, наклейки',
       ' Развитие', ' Сертификаты, услуги', ' Сувениры',
       ' Сувениры (в навеску)', ' Сумки, Альбомы, Коврики д/мыши',
       ' Фигурки', ' 1С:Предприятие 8', ' MAC (Цифра)',
       ' Для дома и офиса', ' Для дома и офиса (Цифра)', ' Обучающие',
       ' Обучающие (Цифра)', 'Служебные', ' Билеты',
       'Чистые носители (шпиль)', 'Чистые носители (штучные)',
       'Элементы питания'], dtype=object)

sub_type_map = dict([(v,k) for k, v in enumerate(cat['sub_type'].unique())])
cat['sub_type_code'] = cat['sub_type'].map(sub_type_map)
cat.head()

items = items.merge(cat[['item_category_id', 'item_type_code', 'sub_type_code']], on='item_category_id', how='left')
items.head()

import gc
del cat
gc.collect()

#必须写x= y= data=
sns.jointplot(x='item_cnt_day', y='item_price', data=train, kind='scatter')
