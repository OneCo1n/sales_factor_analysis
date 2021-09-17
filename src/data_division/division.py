import pandas as pd

# 将单个油站的销售信息分离处理来
def divisionByPlant(df, plant_name):

    plant_class = df['plant'].unique()

    df_temp = df[df['plant'].isin([plant_name])]
    return df_temp

    # locals()
    # print(locals())
    #n = plant_class.size()

    # for one_plant in plant_class:
    #
    #     df[df['plant'].isin([plant_class[0]])]
    #
    #     # #one_plant_name = "df_" + str(one_plant)
    #     temp_df = df[df['plant'].isin([one_plant])]
    #     exec("df%s = temp_df"%one_plant,{'dfAB01':'df%S'%one_plant})








