import numpy as np
import matplotlib.pyplot as plt

def f(x1,x2):
    y=0.5*np.sin(x1)+0.5*np.cos(x2)+0.1*x1+3
    return y
            
def load_data():
    x1_train=np.linspace(0,50,500)
    x2_train=np.linspace(-10,10,500)
    data_train=np.array([[x1,x2,f(x1,x2)+(np.random.random(1)-0.5)]for x1,x2 in zip(x1_train,x2_train)])#加噪声np.random.random(1)-0.5
    x1_test= np.linspace(0,50,100)+0.5*np.random.random(100)
    x2_test= np.linspace(-10,10,100)+0.02*np.random.random(100)
    data_test = np.array([[x1,x2,f(x1,x2)] for x1,x2 in zip(x1_test,x2_test)])
    return data_train, data_test

def main():
    d_train,d_test = load_data()
    plt.subplot()
    plt.scatter(d_train[:,1],d_train[:,2],s=10,color='red')
    plt.scatter(d_test[:,1],d_test[:,2],s=15,color='blue')
    plt.xlabel("time")
    plt.ylabel("data")
    plt.legend(["train data","test data"])
    plt.show()  
    
if __name__=="__main__":
   main()
