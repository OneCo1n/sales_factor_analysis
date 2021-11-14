# use feature importance for feature selection
from numpy import loadtxt
from numpy import sort
from matplotlib import pyplot
from xgboost import XGBClassifier
from xgboost import plot_importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectFromModel
from data_preprocess.columnsExg import *
from compnents.material_plant_pearson import oneMaterialOnAllPlantPearson
from db.dataQuery import *
import matplotlib.pyplot as plt
from db.data_extraction import *
plt.rcParams['font.sans-serif']=['Microsoft YaHei']


#000000000070251989
#000000000070000579
#000000000000400102
def trainMaterialXgboost(material):
    # load data

    df_pearson, dfDsv = oneMaterialOnAllPlantPearson(material)

    # split data into X and y
    print(dfDsv)
    dataset = dfDsv
    # X = dataset[:,1:35]
    # Y = dataset[:,0]
    # split data into train and test sets
    dfDsv = exgColumnsName(dfDsv)
    X_train, X_test, Y_train, Y_test = train_test_split(dfDsv.iloc[:, 1:34], dfDsv.quantity,
                                                        test_size=0.25,train_size=0.75)
    #X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=7)
    # fit model on all training data
    model = XGBClassifier()
    model.fit(X_train, Y_train)
    model.score()
    print(model.feature_importances_)
    # plot
    name = getMaterialName(material)
    pyplot.bar(range(len(model.feature_importances_)), model.feature_importances_)
    plot_importance(model)
    pyplot.show()

#
# y_pred = model.predict(X_test)
# predictions = [round(value) for value in y_pred]
# accuracy = accuracy_score(y_test, predictions)
# print("Accuracy: %.2%%" % (accuracy * 100.0))

# # select features using threshold
# selection = SelectFromModel(model, threshold=thresh, prefit=True)
# select_X_train = selection.transform(X_train)
# # train model
# selection_model = XGBClassifier()
# selection_model.fit(select_X_train, y_train)
# # eval model
# select_X_test = selection.transform(X_test)
# y_pred = selection_model.predict(select_X_test)

# model.fit(X_train, y_train)
# make predictions for test data and evaluate
# y_pred = model.predict(X_test)
# predictions = [round(value) for value in y_pred]
# accuracy = accuracy_score(y_test, predictions)
# print("Accuracy: %.2f%%" % (accuracy * 100.0))
# # Fit model using each importance as a threshold
# thresholds = sort(model.feature_importances_)
#
#
# for thresh in thresholds:
#     # select features using threshold
# 	selection = SelectFromModel(model, threshold=thresh, prefit=True)
# 	select_X_train = selection.transform(X_train)
# 	# train model
# 	selection_model = XGBClassifier()
# 	selection_model.fit(select_X_train, y_train)
# 	# eval model
# 	select_X_test = selection.transform(X_test)
# 	y_pred = selection_model.predict(select_X_test)
# 	predictions = [round(value) for value in y_pred]
# 	accuracy = accuracy_score(y_test, predictions)
# 	print("Thresh=%.3f, n=%d, Accuracy: %.2f%%" % (thresh, select_X_train.shape[1], accuracy*100.0))