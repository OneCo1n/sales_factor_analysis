import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def correlation_analysis(df):
    x_cols = [col for col in df.columns if col not in ['quantity'] if df[col].dtype != 'object']  # 处理目标的其他所有特征
    labels = []
    values = []
    for col in x_cols:
        labels.append(col)
        values.append(np.corrcoef(df[col].values, df.quantity, values)[0, 1])

    corr_df = pd.DataFrame({'col_labels': labels, 'corr_values': values})
    corr_df = corr_df.sort_values(by='corr_values')

    ind = np.arange(len(labels))
    width = 0.5
    fig, ax = plt.subplots(figsize=(12, 40))
    rects = ax.barh(ind, np.array(corr_df.corr_values.values), color='y')

    ax.set_yticks(ind)
    ax.set_yticklabels(corr_df.col_labels.values, rotation='horizontal')
    ax.set_xlabel('Correlation coefficient')
    ax.set_title('Correlation coefficient of the variables')
    plt.show()