import numpy as np 
import matplotlib.pyplot as plt 
from sklearn import linear_model
from sklearn import tree
from sklearn.tree import DecisionTreeRegressor
from sklearn import neighbors
from sklearn import svm
from sklearn import ensemble
from sklearn import neighbors

def f(x1, x2):
    y = 0.5*np.sin(x1)+0.5*np.cos(x2)+0.1*x1+3
    return y

def load_data():
    x1_train = np.linspace(0,50,500)
    x2_train = np.linspace(-10,10,500)
    data_train = np.array([[x1, x2, f(x1,x2)+(np.random.random(1)-0.5)] for x1,x2 in zip(x1_train,x2_train)])
    x1_test = np.linspace(0,50,100) + 0.5*np.random.random(100)
    x2_test = np.linspace(-10,10,100) + 0.02*np.random.random(100)
    data_test = np.array([[x1,x2,f(x1,x2)]for x1,x2 in zip(x1_test, x2_test)])
    return data_train, data_test

def try_different_method(clf):
    train, test = load_data()
    x_train, y_train = train[:,:2], train[:,2]
    x_test, y_test = test[:,:2], test[:,2]
    clf.fit(x_train,y_train)
    score = clf.score(x_test, y_test)
    result = clf.predict(x_test)
    plt.figure()
    plt.plot(np.arange(len(result)), y_test, 'go-', label='true value')
    plt.plot(np.arange(len(result)), result, 'ro-', label='predict value')
    plt.title('score:%f'%score)
    plt.legend()
    plt.show()

def main():
    # clf = linear_model.LinearRegression()
    # try_different_method(clf)
    # clf = DecisionTreeRegressor()
    # try_different_method(clf)
    # clf = svm.SVR()
    # try_different_method(clf)
    # clf = neighbors.KNeighborsRegressor()
    # try_different_method(clf)
    # clf = ensemble.RandomForestRegressor()
    # try_different_method(clf)
    # clf = ensemble.AdaBoostRegressor()
    # try_different_method(clf)
    clf = ensemble.GradientBoostingRegressor()
    try_different_method(clf)

if __name__ == '__main__':
    main()