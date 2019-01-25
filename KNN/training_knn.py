from util import load_all_cifar, showimages, time_elapse
from k_nearest_neighbor import KNearestNeighbor
import matplotlib.pyplot as plt
import numpy as np

root = 'cifar-10-batches-py'
# load data
X_train, y_train, X_test, y_test = load_all_cifar(root)

# showimages(X_train, y_train, 10)

# subsample data
num_training = 5000
mask = range(num_training)
# subsample train data. array indexing
X_train = X_train[mask]
y_train = y_train[mask]

num_test = 500
mask = range(num_test)
# subsample test data. array indexing
X_test = X_test[mask]
y_test = y_test[mask]

# reshape data to rows
X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))

# initial classifer object
knn_classifer = KNearestNeighbor()
# train with training set
# knn_classifer.train(X_train, y_train)

"""
# compute distances for test set
"""
# dists = knn_classifer.compute_distances_two_loops(X_test)
# dists_one_loop = knn_classifer.compute_distances_one_loop(X_test)
# dists_no_loop = knn_classifer.compute_distances_no_loops(X_test)

"""
# test one loop computation
"""
# difference = np.linalg.norm(dists - dists_one, ord='fro')
# print('One loop Difference was: %f' % (difference, ))
"""
test no loop computation
"""
# difference = np.linalg.norm(dists - dists_no, ord='fro')
# print('No loop Difference was: %f' % (difference, ))

"""
# test running times

two_loop_time = time_elapse(knn_classifer.compute_distances_two_loops, X_test)
print('Two loop version took %f seconds' % two_loop_time)

one_loop_time = time_elapse(knn_classifer.compute_distances_one_loop, X_test)
print('One loop version took %f seconds' % one_loop_time)

no_loop_time = time_elapse(knn_classifer.compute_distances_no_loops, X_test)
print('No loop version took %f seconds' % no_loop_time)
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
"""
"""
# test k=5
k = 5
y_test_pred = knn_classifer.predict_labels(dists_no, k)
num_correct = np.sum(y_test_pred == y_test)
accuracy = num_correct / num_test
print('Got %d / %d correct => accuracy: %f with k = %d' %
      (num_correct, num_test, accuracy, k))
"""

"""
k fold Cross validation
"""
# 5 fold cross valdiation
num_folds = 5
# a list of k choices
k_choices = [1, 3, 5, 8, 10, 12, 15, 20, 50, 100]

X_train_folds = np.array(np.split(X_train, num_folds))
y_train_folds = np.array(np.split(y_train, num_folds))
k_to_accuracies = {}
# test each k
for k in k_choices:
    # loop for each validation fold
    for val_idx in range(num_folds):
        # get a list of indexes of training folds, e.g. [1,2,3,4] [0,2,3,4]
        train_idx = [i for i in range(num_folds) if i != val_idx]
        # get training set x & y
        X_train_set = np.concatenate(X_train_folds[train_idx])
        y_train_set = np.concatenate(y_train_folds[train_idx])
        # train
        knn_classifer.train(X_train_set, y_train_set)
        # get prediction with current validation fold
        predict_y = knn_classifer.predict(X_train_folds[val_idx], k)
        # compute acc for the current validation fold
        accuracy = np.mean(predict_y == y_train_folds[val_idx])
        # store the accuracy
        k_to_accuracies.setdefault(k, []).append(accuracy)

# print out the computed accuracies
for k in sorted(k_to_accuracies):
    for accuracy in k_to_accuracies[k]:
        print('k = %d is %f' % (k, accuracy))
    print('mean for k = %d is %f' % (k, np.mean(k_to_accuracies[k])))
# plot
for k in k_choices:
    accuracies = k_to_accuracies[k]
    plt.scatter([k] * len(accuracies), accuracies)

# plot the trend line with error bars that correspond to standard deviation
accuracies_mean = np.array([np.mean(v)
                            for k, v in sorted(k_to_accuracies.items())])
accuracies_std = np.array([np.std(v)
                           for k, v in sorted(k_to_accuracies.items())])
plt.errorbar(k_choices, accuracies_mean, yerr=accuracies_std)
plt.title('Cross-validation on k')
plt.xlabel('k')
plt.ylabel('Cross-validation accuracy')
plt.show()

k = 10
new_classifier = KNearestNeighbor()
new_classifier.train(X_train, y_train)
y_test_predictions = new_classifier.predict(X_test, k)
num_correct = np.sum(y_test_predictions == y_test)
accuracy = num_correct / num_test
#accuracy = np.mean(y_test_predictions == y_test)
print('Got %d / %d correct => accuracy: %f' %
      (num_correct, num_test, accuracy))
