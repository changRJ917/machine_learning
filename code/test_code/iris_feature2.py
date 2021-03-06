#import tensorflow as tf
from sklearn.datasets import load_iris
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression


iris = load_iris()

selector = RFE(estimator=LogisticRegression(), n_features_to_select=1).fit(iris.data, iris.target)
data = selector.transform(iris.data)
#print(selector.n_features_)
#print(data)
print(selector.ranking_)
