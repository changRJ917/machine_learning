from sklearn.datasets import load_iris
from sklearn import tree

iris = load_iris()
clf = tree.DecisionTreeClassifier()
clf = clf.fit(iris.data, iris.target)
print(clf)
from sklearn.externals import joblib
joblib.dump(clf, 'train_model.m')
clf2 = joblib.load('train_model.m')
pred = clf2.predict(iris.data[:])
pred_prob = clf2.predict_proba(iris.data[:1, :])


import graphviz
dot_data = tree.export_graphviz(clf, out_file=None)
graph = graphviz.Source(dot_data)
graph.render('iris')
