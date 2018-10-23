from sklearn.datasets import load_iris
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
iris = load_iris()
X = iris.data
y = iris.target

param_grid = [
   {'n_estimators':[3,10,30],'max_features':[1,2,3,4]},
   {'bootstrap':[False],'n_estimators':[3,10],'max_features':[2,3,4]},
]

forest_reg = RandomForestRegressor()
grid_search = GridSearchCV(forest_reg,param_grid,scoring='mean_squared_error')
grid_search.fit(X,y)

print(grid_search.best_score_)
print(grid_search.best_params_)

print(grid_search.best_estimator_)