from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder()
enc.fit([[0, 0, 3], [1, 1, 0], [0, 2, 1], [1, 0, 2]])
#一个4行3列的数组，即有4个样本，3个特征；
print(enc.n_values_)
#每个特征的取值个数（第一列共有2个取值，第二列有3个，第三列有4个）
#输出：array([2, 3, 4])
print(enc.feature_indices_)
# print(enc.feature_indices_)
#独热编码后每个特征开始和结束的下标；
#输出：array([0, 2, 5, 9])

print(enc.transform([[1, 2, 1]]).toarray())
#对新拿到的[0,1,1]做独热编码，即有第一列是[1,0],第二列是[0,1,0]
#第三列是[0,1,0,0]，所以最终拼在一起得到它的独热编码就是[1,0,0,1,0,0,1,0,0]
#output:array([[ 1.,  0.,  0.,  1.,  0.,  0.,  1.,  0.,  0.]])

