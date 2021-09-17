def replace_outlier(val, mean, std):
    if val > mean + 3*std:
        return mean + 3*std
    elif val < mean - 3*std:
        return mean - 3*std
    return val


def outlierProcessing(df):
    cols = ["quantity","number_station","number_store","building_area","business_hall","paking_area","store_area","discount_rate2",]
    for col in cols:
        mean = df[col].mean()
        std_dev = df[col].std(axis=0)
        df[col] = df[col].map(lambda x: replace_outlier(x, mean, std_dev))