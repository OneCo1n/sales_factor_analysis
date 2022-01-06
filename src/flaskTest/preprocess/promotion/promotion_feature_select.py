

def promotion_feature_select(promotion):

    promotion = promotion.drop(labels=None, axis=1, index=None, columns=['discount', 'promotion_quantity', 'promotion_amount', 'retail_price'], inplace=False)
    promotion = promotion.drop_duplicates()

    return promotion