import numpy as np

# 处理含有负数的记录
def promotion_discount_rate_outliers_replace(promotion):
    # df_data_material_promotion = df_data_material_promotion.abs()
    promotion['discount'] = promotion['discount'].astype('float64')
    promotion['promotion_amount'] = promotion['promotion_amount'].astype('float64')
    promotion['retail_price'] = promotion['retail_price'].astype('float64')
    promotion['discount' ] = promotion['discount'].abs()
    promotion['promotion_amount'] = promotion['promotion_amount'].abs()
    promotion['retail_price'] = promotion['retail_price'].abs()
    promotion['promotion_amount'], promotion['retail_price'] = np.where(promotion['promotion_amount'] > promotion['retail_price'],
                                                                        [promotion['retail_price'], promotion['promotion_amount']],
                                                                        [promotion['promotion_amount'], promotion['retail_price']])
    promotion['discount_rate2'] = (promotion['promotion_amount'] + 0.00001) / (promotion['retail_price'] + 0.00001)
    promotion[(promotion['discount_rate2'] < 0) & (promotion['discount_rate2'] > 1)]

    return promotion



