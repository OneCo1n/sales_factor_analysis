from matplotlib import pyplot

#绘制箱形图
def drawBox(df):
    #创建箱形图
    #第一个参数为待绘制的定量数据
    #第二个参数为数据的文字说明
    pyplot.boxplot([df.quantity], labels=['quantity'])
    pyplot.title('sales of material')
    pyplot.show()

