import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

def pairplot(df, material):
    sns.pairplot(df, x_vars=["number_station","number_store","building_area","business_hall","paking_area","store_area","discount_rate2","plant_asset", "road_class", "plant_class_code", "plant_location_class", "promotion_type", "plant_type_desc"], y_vars='quantity', height=7, aspect=0.8, kind='reg')
    plt.savefig("pairplot" + material+ ".jpg")
    plt.show()