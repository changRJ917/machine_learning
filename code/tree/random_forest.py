from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

X,y = make_classification(n_samples=1000, n_features=4, n_informative=2, n_redundant=0,
random_state=0, shuffle=False,)
clf = RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(X, y)
print(clf.feature_importances_)
print(clf.predict([[0,0,0,0]]))

###
from sklearn.tree import DecisionTreeClassifier
import numpy as np 
from sklearn.datasets import load_iris
iris = load_iris()
rf = RandomForestClassifier()
rf.fit(iris.data, iris.target)

instance = iris.data[[100,109]]
print(instance)
print(rf.predict(instance[[0]]))
print(iris.target[[100,109]], iris.target[109])
####
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_blobs
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
X,y = make_blobs(n_samples=10000, n_features=100, centers=100, random_state=0)
print(X[:1], y[:1])