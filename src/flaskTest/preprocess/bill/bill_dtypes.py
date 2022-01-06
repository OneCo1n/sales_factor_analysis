

def bill_dtypes_exg(bill):

    bill['plant'] = bill['plant'].astype('str')
    bill['date'] = bill['date'].astype('datetime64')
    # bill['date'] = bill['date'].astype('str')
    bill['quantity'] = bill['quantity'].astype('int')

    return bill