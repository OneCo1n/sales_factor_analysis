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


def LabelEncoder_asset(df):
    col = "plant_asset"
    df.loc[df[col] == '0001', col] = 1;
    df.loc[df[col] == '0002', col] = 2;
    df.loc[df[col] == '0003', col] = 3;
    df.loc[df[col] == '0004', col] = 4;
    df[col] = df[col].astype('int')


def LabelEncoder_roadClass(df):
    col = "road_class"
    df.loc[df[col] == '0001', col] = 1;
    df.loc[df[col] == '0002', col] = 2;
    df.loc[df[col] == '0003', col] = 3;
    df.loc[df[col] == '0004', col] = 4;
    df[col] = df[col].astype('int')

def LabelEncoder_plantClass(df):
    col = "plant_class_code"
    df.loc[df[col] == '0001', col] = 1;
    df.loc[df[col] == '0002', col] = 2;
    df.loc[df[col] == '0003', col] = 3;
    df.loc[df[col] == '0004', col] = 4;
    df[col] = df[col].astype('int')

def LabelEncoder_locationClass(df):
    col = "plant_location_class"
    df.loc[df[col] == '001', col] = 1;
    df.loc[df[col] == '002', col] = 2;
    df.loc[df[col] == '003', col] = 3;
    df.loc[df[col] == '004', col] = 4;
    df.loc[df[col] == '005', col] = 5;
    df.loc[df[col] == '006', col] = 6;
    df.loc[df[col] == '007', col] = 7;
    df.loc[df[col] == '008', col] = 8;
    df.loc[df[col] == '009', col] = 9;
    df.loc[df[col] == '010', col] = 10;
    df.loc[df[col] == '011', col] = 11;
    df.loc[df[col] == '012', col] = 12;
    df[col] = df[col].astype('int')


def LabelEncoder_promotionType(df):
    col = "promotion_type"
    df.loc[df[col] == 'NP', col] = 1;
    df.loc[df[col] == 'C', col] = 2;
    df.loc[df[col] == 'O', col] = 3;
    df.loc[df[col] == 'A', col] = 4;
    df.loc[df[col] == 'G', col] = 5;
    df.loc[df[col] == 'BB01', col] = 6;
    df.loc[df[col] == 'BB02', col] = 7;
    df.loc[df[col] == 'BB03', col] = 8;
    df.loc[df[col] == 'BB04', col] = 9;
    df.loc[df[col] == 'BB05', col] = 10;
    df.loc[df[col] == 'BB06', col] = 11;
    df.loc[df[col] == 'BB07', col] = 12;
    df.loc[df[col] == 'BB08', col] = 13;
    df.loc[df[col] == 'BB09', col] = 14;
    df.loc[df[col] == 'BB10', col] = 15;
    df[col] = df[col].astype('int')

def LabelEncoder_plantType(df):
    col = "plant_type_desc"
    df.loc[df[col] == '普通站', col] = 1;
    df.loc[df[col] == '纯便利店', col] = 2;
    df.loc[df[col] == 'D类站', col] = 3;
    df.loc[df[col] == '中转仓', col] = 4;
    df.loc[df[col] == '虚拟站', col] = 5;
    df.loc[df[col] == '中央仓', col] = 6;
    df[col] = df[col].astype('int')





