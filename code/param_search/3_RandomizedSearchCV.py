from sklearn.datasets import load_iris
from sklearn.grid_search import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor

iris = load_iris()
X = iris.data
y = iris.target

param_distribs = {
    'n_estimators': range(1,100),
    'max_features': range(1,4)
}

forest_reg = RandomForestRegressor()
rnd_search = RandomizedSearchCV(forest_reg, param_distribs, n_iter=20)
rnd_search.fit(X,y)

print(rnd_search.best_score_)
print(rnd_search.best_params_)

print(rnd_search.best_estimator_)