

def discount_dtypes_exg(discount):
    # discount.columns=('plant', 'calday', 'discount_rate')
    # discount['plant'] = discount['plant'].astype('str')
    discount['date'] = discount['date'].astype('datetime64')
    discount['date'] = discount['date'].astype('str')

    discount['discount_rate'] = discount['discount_rate'].astype('float64')

    return discount