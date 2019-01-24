import pickle
import os
import numpy as np


def load_one_cifar_batch(file):
    with open(file, 'rb') as file:
        dict = pickle.load(file, encoding='bytes')
        X = dict[b'data']
        Y = dict[b'labels']
        X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
        Y = np.array(Y)
    return X, Y


def load_all_cifar(root):
    x_all = []
    y_all = []
    for i in range(1, 6):
        filepath = os.path.join(root, "data_batch_%d" % (i,))
        X, Y = load_one_cifar_batch(filepath)
        x_all.append(X)
        y_all.append(Y)
    Xtrain = np.concatenate(x_all)
    Ytrain = np.concatenate(y_all)
    del X, Y
    Xtest, Ytest = load_one_cifar_batch(os.path.join(root, "test_batch"))
    return Xtrain, Ytrain, Xtest, Ytest
