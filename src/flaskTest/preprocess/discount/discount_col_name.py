

# 更改discount的列明
def discount_col_name_exg(discount):

    discount.rename(columns={'calday': 'date'}, inplace=True)

    return discount


