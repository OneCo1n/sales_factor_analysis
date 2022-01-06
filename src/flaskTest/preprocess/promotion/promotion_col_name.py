

# 更改promotion的列明
def promotion_col_name_exg(promotion):

    promotion.rename(columns={'calday': 'date'}, inplace=True)

    return promotion