from __future__ import print_function 
from sklearn.datasets import load_iris
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Binarizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2#选择K个最好的特征，返回选择特征后的数据
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import GradientBoostingClassifier 
iris = load_iris()
########（1）标准化，返回值为标准化后的数据
#X = StandardScaler().fit_transform(iris.data)
########（2）区间缩放，返回值为缩放到[0, 1]区间的数据
X1 = MinMaxScaler().fit_transform(iris.data)
########（3）二值化，阈值设置为3，返回值为二值化后的数据
#X = Binarizer(threshold=3).fit_transform(iris.data)
#哑编码，对IRIS数据集的目标值，返回值为哑编码后的数据(不知道怎么搞？？？？)
# OneHotEncoder().fit_transform(iris.target.reshape((-1,1)))
#X = iris.data

# selector = SelectKBest(chi2, k=2).fit(X1, iris.target)
# X = selector.transform(X1)

# selector = RFE(estimator=LogisticRegression(), n_features_to_select=2).fit(X1, iris.target)
# X = selector.transform(X1)

selector = SelectFromModel(GradientBoostingClassifier()).fit(X1, iris.target)
X = selector.transform(X1)
y = iris.target



print(X, y)
# (0) feature engineering
# the degree-2 polynomial features are [1, a, b, a^2, ab, b^2].
poly = PolynomialFeatures(2)
X_Poly = poly.fit_transform(X)
###X = X_Poly
print(X_Poly)

# (1) test train split #
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=4)
knn = KNeighborsClassifier(n_neighbors=5)

# (2) Model training
knn.fit(X_train, y_train)

# (3) Predict & Estimate the score
# y_pred = knn.predict(X_test)
# print(knn.score(X_test, y_test))
y_pred = knn.predict(X_train)
print(knn.score(X_train, y_train))

