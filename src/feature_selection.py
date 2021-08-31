import warnings
warnings.filterwarnings("ignore")

# 方差法
from sklearn.feature_selection import VarianceThreshold
X = [[0, 2, 0, 3], [0, 1, 4, 3], [0, 1, 1, 3]]
selector = VarianceThreshold()
X_new = selector.fit_transform(X)
#
print('X_new:\n',selector.fit_transform(X))
print('get_params:\n',selector.get_params())
print('get_support:\n',selector.get_support())
print('inverse_transform:\n',selector.inverse_transform(X_new))