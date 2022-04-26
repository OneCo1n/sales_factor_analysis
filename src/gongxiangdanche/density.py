import matplotlib.pyplot as plt
import seaborn as sns

def show_density(df, col):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    fig.set_size_inches(6, 5)
    ax = sns.distplot(df[col])  # will be removed in a future version
    # ax = sns.histplot(train['count'])
    # sns.displot(train['count'])  # 这种写法 子图和画布会分离
    ax.set(xlabel='非油品日销量', title= '非油品日销量密度分布')
    plt.savefig('非油品日销量密度分布.png', bbox_inches='tight')
    plt.show()


def show_discount_density(df, col):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    fig.set_size_inches(6, 5)
    ax = sns.distplot(df[col])  # will be removed in a future version
    # ax = sns.histplot(train['count'])
    # sns.displot(train['count'])  # 这种写法 子图和画布会分离
    ax.set(xlabel='非油品折扣', title= '非油品折扣密度分布')
    plt.savefig('非油品折扣密度分布.png', bbox_inches='tight')
    plt.show()