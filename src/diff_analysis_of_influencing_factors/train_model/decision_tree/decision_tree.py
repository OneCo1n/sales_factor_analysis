from sklearn.datasets import load_iris

from sklearn import tree

# 引入数据

iris = load_iris()

X = iris.data

y = iris.target

# 训练数据和模型,采用ID3或C4.5训练

clf = tree.DecisionTreeClassifier(criterion='entropy')

clf = clf.fit(X, y)

# 引入graphviz模块用来导出图,结果图如下所示

import graphviz

dot_data = tree.export_graphviz(clf, out_file=None, feature_names=iris.feature_names,
    class_names=iris.target_names,filled=True,rounded=True,special_characters=True
)

graph=graphviz.Source(dot_data)

graph.view()