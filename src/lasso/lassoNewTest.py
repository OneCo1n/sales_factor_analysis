# Make the necessary imports
from sklearn.linear_model import LassoCV

# Load the dataset

from flaskTest.views.data_views.total_views import get_total_by_material

material = '000000000070251989'

data = get_total_by_material('000000000070251989')

def features_select_use_lasso(data):

    predictors = data['quantity']
    data = data.iloc[:, 0:]

    target = data.quantity

    features = data.drop(labels=None, axis=1, index=None, columns=['plant', 'date', 'quantity'], inplace=False)

    regressor=LassoCV()
    regressor.fit(features,target)
    coef_col = (regressor.coef_)

    coef_col_list = list(coef_col)
    features_list = list(features.columns)
    # print(coef_col_list)
    # print(features_list)

    features_coef_dict =  dict()

    for i in range(len(coef_col_list)):
        features_coef_dict[features_list[i]] = coef_col_list[i]

    print(features_coef_dict)

    selected_features =(features.columns[(coef_col != 0)])
    print(selected_features)

    rejected_features =  features.columns[(coef_col == 0)]
    print(rejected_features)

    return list(rejected_features)

#Drop rejected features.
#Standardize new features.


