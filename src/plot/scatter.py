from matplotlib import pyplot

#绘制散点图
def drawScatter(df,x,y):
    #创建散点图
    #第一个参数为点的横坐标
    #第二个参数为点的纵坐标
    pyplot.scatter(df[x], df[y])
    pyplot.xlabel(x)
    pyplot.ylabel(y)
    pyplot.title(x + ' & ' + y +'  scatter')
    pyplot.show()

