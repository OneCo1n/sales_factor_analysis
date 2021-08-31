import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import csv
from pandas.core.frame import DataFrame


def minMaxScaler(df):
    scaler = MinMaxScaler()
    scaler.fit(df)
    scaled_features = scaler.transform(df)
    df_MinMax = pd.DataFrame(data=scaled_features)







tmp_lst = []
with open('F:\\论文\\data\\df_encoding.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        tmp_lst.append(row)
df = pd.DataFrame(tmp_lst[1:], columns=tmp_lst[0])
print(df)
df = df.drop("plant",axis= 1)
df = df.drop("date",axis = 1)

minMaxScaler(df)
print(df)

outputpath='F:\论文\data\\biaozhunhua.csv'
df.to_csv(outputpath,index=True,header=True)