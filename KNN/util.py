import pickle
import os
import numpy as np
import time
import matplotlib.pyplot as plt


def load_one_cifar_batch(file):
    """
    load one batch training set
    """
    with open(file, 'rb') as file:
        dict = pickle.load(file, encoding='bytes')
        X = dict[b'data']
        Y = dict[b'labels']
        X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
        Y = np.array(Y)
    return X, Y


def load_all_cifar(root):
    """
    load 5 batches training set and 1 test set
    """
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


def showimages(X_train, y_train, sample_per_class=8):
    # define class name list
    class_list = ['plane', 'car', 'bird', 'cat',
                  'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    num_classes = len(class_list)
    # print some pictures from training set
    for class_index, class_name in enumerate(class_list):
        # get indexes in the label list that are equal to the index of the class list
        y_train_indexes = np.flatnonzero(y_train == class_index)
        # randomly pick sample indexes from the class
        y_train_indexes = np.random.choice(
            y_train_indexes, sample_per_class, replace=False)
        # show images
        for i, y_index in enumerate(y_train_indexes):
            plt_idx = i * num_classes + class_index + 1
            plt.subplot(sample_per_class, num_classes, plt_idx)
            plt.imshow(X_train[y_index].astype('uint8'))
            plt.axis('off')
            if i == 0:
                plt.title(class_name)
    plt.show()


def time_elapse(function, *args):
    """
    Call a function with args and return the time (in seconds) that it took to execute.
    """
    tic = time.time()
    function(*args)
    toc = time.time()
    return toc - tic
