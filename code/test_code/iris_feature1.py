import tensorflow as tf
from sklearn.datasets import load_iris

iris = load_iris()

data1 = tf.constant(iris.data[:,0])
data2 = tf.constant(iris.data[:,1])
with tf.Session() as sess:
    result = sess.run(data1*data2)
    print(result)

