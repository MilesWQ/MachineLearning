from loadcifar.loadfile import load_all_cifar
from matplotlib import pyplot as plt
from displayimage import showimages
from k_nearest_neighbor import KNearestNeighbor
import numpy as np

root = 'cifar-10-batches-py'
# load data
X_train, y_train, X_test, y_test = load_all_cifar(root)

#showimages(X_train, y_train, 10)

# subsample data
num_training = 5000
mask = range(num_training)
# array indexing
X_train = X_train[mask]
y_train = y_train[mask]

num_test = 500
mask = range(num_test)
# array indexing
X_test = X_test[mask]
y_test = y_test[mask]

# reshape data to rows
X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))

# initial classifer object
knn_classifer = KNearestNeighbor()
# train with training set
knn_classifer.train(X_train, y_train)

# compute distances for test set
#dists = knn_classifer.compute_distances_two_loops(X_test)
"""
test one loop computation
dists_one = knn_classifer.compute_distances_one_loop(X_test)
difference = np.linalg.norm(dists - dists_one, ord='fro')
print('Difference was: %f' % (difference, ))
"""

"""
# test k=1
k = 1
y_test_pred = knn_classifer.predict_labels(dists, k)
# count correct predictions
num_correct = np.sum(y_test_pred == y_test)
accuracy = num_correct / num_test
print('Got %d / %d correct => accuracy: %f with k = %d' %
      (num_correct, num_test, accuracy, k))

# test k=5
k = 5
y_test_pred = knn_classifer.predict_labels(dists, k)
num_correct = np.sum(y_test_pred == y_test)
accuracy = num_correct / num_test
print('Got %d / %d correct => accuracy: %f with k = %d' %
      (num_correct, num_test, accuracy, k))
"""

a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
test = np.array([[1, 1, 1], [2, 2, 2]])
b = a.reshape(1, a.size)
print(b)
