

def promotion_dtypes_exg(promotion):
    # promotion.columns=('plant', 'promotion_type', 'discount', 'promotion_quantity', 'promotion_amount', 'retail_price', 'calday' )
    promotion['plant'] = promotion['plant'].astype('str')
    promotion['promotion_type'] = promotion['promotion_type'].astype('str')
    promotion['discount'] = promotion['discount'].astype('float64')
    promotion['promotion_quantity'] = promotion['promotion_quantity'].astype('float64')
    promotion['promotion_quantity'] = promotion['promotion_quantity'].astype('int')
    promotion['promotion_amount'] = promotion['promotion_amount'].astype('float64')
    promotion['retail_price'] = promotion['retail_price'].astype('float64')
    promotion['date'] = promotion['date'].astype('datetime64')
    # promotion['date'] = promotion['date'].astype('str')


    return promotion