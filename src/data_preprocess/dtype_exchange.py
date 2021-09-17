


def date_conversion_toStr(df):

    for i in df.columns[1:]:
        df[i] = df[i].astype('str')

    return df


def pearsonDytypeExg(df):

    #df['material'] = df['material'].astype('str')
    df['plant'] = df['plant'].astype('str')

    for col in df.columns[2:]:
        df[col] = df[col].astype('float64')

    return df
