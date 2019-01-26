import numpy as np


X = np.array([[0, 2, 3, 0], [3, 0, 4, 1], [
             2, 5, 8, 7], [1, 8, 9, 1], [7, 6, 9, 4]])
X_1 = np.array([0, 2, 3, 0])
# W DxC W.t shape = CxD c
W = np.array([[0.1, 0.2, 1], [0.2, 0.3, 0.4], [
             0.5, 0.6, 0.7], [0.7, 0.8, 0.9]])
# x_predict = D x N
X_predict = X.T
scores = W.T.dot(X_predict)
# print(scores)
scores = np.argmax(scores, axis=0)
# print(scores)

r = [2, 10]

for i in r:
    print(i)
