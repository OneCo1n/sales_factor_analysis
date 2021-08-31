from pandas import Series
from sklearn import preprocessing
import pandas as pd


def LabelEncoder_stars(df, col):

    df.loc[df[col] == '11', col] = 0;
    df.loc[df[col] == '10', col] = 1;
    df.loc[df[col] == '01', col] = 2;
    df.loc[df[col] == '02', col] = 3;
    df.loc[df[col] == '03', col] = 4;
    df.loc[df[col] == '04', col] = 5;
    df.loc[df[col] == '05', col] = 6;
    df[col] = df[col].astype('int')



def LabelEncoder_keyanliang(df, col):

    df.loc[df[col] == '年销量500-1000T', col] = 1;
    df.loc[df[col] == '年销量1000-3000T', col] = 2;
    df.loc[df[col] == '年销量3000-5000T', col] = 3;
    df.loc[df[col] == '年销量5000-10000T', col] = 4;
    df.loc[df[col] == '年销量10000T以上', col] = 5;
    df[col] = df[col].astype('int')









