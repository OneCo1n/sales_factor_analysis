import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn.linear_model import Lasso,LassoCV
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from math import sqrt
from sklearn.metrics import r2_score#R square

# Import helper functions


from compnents.material_plant_pearson import oneMaterialOnAllPlantPearson
from db import data_extraction

material = '000000000070309748'
#data = data_extraction.get_df_from_db(material)
pearson, data = oneMaterialOnAllPlantPearson(material)
print(data)
#data=pd.read_excel(r'C:\Users\Administrator\Desktop\diabetes.xlsx')
#predictors=data.columns[:-1]
predictors = data['quantity']
data = data.iloc[:, 0:]
#data=data.drop(['AGE','SEX'],axis=1)
#拆分为训练集和测试集

x_train,x_test,y_train,y_test=model_selection.train_test_split(data.iloc[:, 1:34], data.quantity,
                                                        test_size=0.25,train_size=0.75)

# x_train,x_test,y_train,y_test=model_selection.train_test_split(data[predictors],data.Y,
#                                                                test_size=0.2,random_state=1234)
#构造不同的lambda值
Lambdas=np.logspace(-5,2,200)
#设置交叉验证的参数，使用均方误差评估
lasso_cv=LassoCV(alphas=Lambdas,normalize=True,cv=10,max_iter=10000)
lasso_cv.fit(x_train,y_train)

#基于最佳lambda值建模
lasso=Lasso(alpha=lasso_cv.alpha_,normalize=True,max_iter=10000)
lasso.fit(x_train,y_train)
#打印回归系数

reg_coef = pd.Series(index=['Intercept']+x_train.columns.tolist(),
                data=[lasso.intercept_]+lasso.coef_.tolist())

print(reg_coef)

#模型评估
lasso_pred=lasso.predict(x_test)
#均方误差
MSE=mean_squared_error(y_test,lasso_pred)
rmse = sqrt(MSE)
print("rmse" , rmse)
print("r2" ,r2_score(y_test, lasso_pred))


