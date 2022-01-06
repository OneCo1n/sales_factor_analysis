import matplotlib.pyplot as plt
import seaborn as sns

def show_density(df, col):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    fig.set_size_inches(6, 5)
    ax = sns.distplot(df[col])  # will be removed in a future version
    # ax = sns.histplot(train['count'])
    # sns.displot(train['count'])  # 这种写法 子图和画布会分离
    ax.set(xlabel=col, title= col + '密度分布')
    plt.show()