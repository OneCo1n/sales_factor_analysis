from flaskTest.db.bill import queryDailySalesByMaterialAndPlant
import pandas as pd

def getDailySalesByMaterialAndPlant(material, plants):

    # 从数据库获取销量序列
    pCount = len(plants)

    dailySales = pd.DataFrame(columns=['日期'])

    for plant in plants:
        df = queryDailySalesByMaterialAndPlant(material, plant)
        df.rename(columns={'quantity': plant}, inplace=True)
        df.rename(columns={'date': '日期'}, inplace=True)
        df.drop(labels=None, axis=1, index=None, columns=['plant'], inplace=True)
        dailySales = pd.merge(dailySales, df, on=['日期'], how='outer')


    print(dailySales)
    print(dailySales.dtypes)
    print(dailySales.columns)

    return dailySales
plant = {'AK0I', 'AK0M', 'AJ0N'}
getDailySalesByMaterialAndPlant('000000000070251989', plant)






