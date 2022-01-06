import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def log_transform(df, col):
    yLabels = df[col]
    yLabels_log = np.log(yLabels)
    sns.distplot(yLabels_log)
    plt.show()