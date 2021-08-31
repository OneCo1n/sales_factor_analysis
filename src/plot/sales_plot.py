import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates
import pandas as pd
#from matplotlib.dates import bytespdate2num

def sales_plots_byPlant(df,plant):

    df_temp = df[df['plant'].isin([plant])]
# Fixing random state for reproducibility
    fig = plt.figure()
    fig.suptitle('figure title demo', fontsize = 14, fontweight = 'bold')
    ax = fig.add_subplot(1,1,1)
    ax.set_xlabel("x label")

    ax.set_ylabel("y label")

    dates, quantity = df_temp['date'],df_temp['quantity']

    ax.xaxis.set_major_locator(mdates.DayLocator(bymonthday=range(1, 32), interval=3))

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))

    #plt.xticks(pd.date_range('2018-06-19', '2018-07-1', freq='3D'), rotation=25)

    ax.plot(dates, quantity)

    plt.show()


def sales_plots_allPlant(df):

    plants = df['plant'].unique()


    # Fixing random state for reproducibility
    fig = plt.figure()
    fig.suptitle('figure title demo', fontsize=14, fontweight='bold')
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel("x label")

    ax.set_ylabel("y label")

    ax.xaxis.set_major_locator(mdates.DayLocator(bymonthday=range(1, 32), interval=3))

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))

    # plt.xticks(pd.date_range('2018-06-19', '2018-07-1', freq='3D'), rotation=25)
    for plant in plants:
        df_temp = df[df['plant'].isin([plant])]
        dates, quantity = df_temp['date'], df_temp['quantity']
        ax.plot(dates, quantity)

    plt.show()