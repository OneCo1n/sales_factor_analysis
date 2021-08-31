import  pandas as pd
import numpy as np
# # 散点图矩阵初判多变量间关系
# data = pd.DataFrame(np.random.randn(200,4)*100, columns = ['A','B','C','D'])
# pd.plotting.scatter_matrix(data,figsize=(8,8),
#                          c = 'k',
#                          marker = '+',
#                          diagonal='hist',
#                          alpha = 0.8,
#                          range_padding=0.1)
# data.head()

n = 2
for i in range(n):
    exec("temp%s=1"%i)
print("temp0=%d,temp1=%d" %(temp0,temp1))