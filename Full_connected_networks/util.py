import pickle
import os
import numpy as np
import time
import matplotlib.pyplot as plt
from math import sqrt, ceil
from random import randrange


def rel_error(x, y):
    """ returns relative error """
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))


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
    """
    plot images in category
    """
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


def load_data(dir, num_training=49000, num_validation=1000, num_test=1000, num_dev=500, bias=False, row=False):
    """
    Load cifar data and preprocess data
    """
    # load data
    X_train, y_train, X_test, y_test = load_all_cifar(dir)
    # get the validation set
    mask = range(num_training, num_training + num_validation)
    X_val = X_train[mask]
    y_val = y_train[mask]
    # get the training set
    mask = range(num_training)
    X_train = X_train[mask]
    y_train = y_train[mask]
    # get a development set as the subset of training set
    mask = range(num_dev)
    X_dev = X_train[mask]
    y_dev = y_train[mask]
    # get a sub test set
    mask = range(num_test)
    X_test = X_test[mask]
    y_test = y_test[mask]
    """
    preprocess data
    """
    # Mean substraction
    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image
    X_dev -= mean_image
    """
    # visualize the mean image
    """
    # plt.figure(figsize=(4, 4))
    # plt.imshow(mean_image.reshape((32, 32, 3)).astype('uint8'))
    # plt.show()
    # reshape the image data into rows
    if row:
        X_train = np.reshape(X_train, (X_train.shape[0], -1))
        X_val = np.reshape(X_val, (X_val.shape[0], -1))
        X_test = np.reshape(X_test, (X_test.shape[0], -1))
        X_dev = np.reshape(X_dev, (X_dev.shape[0], -1))
        # if need to add a bias 1 to the last column
        if bias:
            X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
            X_val = np.hstack([X_val, np.ones((X_val.shape[0], 1))])
            X_test = np.hstack([X_test, np.ones((X_test.shape[0], 1))])
            X_dev = np.hstack([X_dev, np.ones((X_dev.shape[0], 1))])
    else:
        # Retain the input shape and change the channel to the first
        X_train = X_train.transpose(0, 3, 1, 2).copy()
        X_val = X_val.transpose(0, 3, 1, 2).copy()
        X_test = X_test.transpose(0, 3, 1, 2).copy()
        X_dev = X_dev.transpose(0, 3, 1, 2).copy()

    print('Train data shape: ', X_train.shape)
    print('Train labels shape: ', y_train.shape)
    print('Validation data shape: ', X_val.shape)
    print('Validation labels shape: ', y_val.shape)
    print('Test data shape: ', X_test.shape)
    print('Test labels shape: ', y_test.shape)
    return X_train, y_train, X_val, y_val, X_test, y_test, X_dev, y_dev


def eval_numerical_gradient(f, x, verbose=True, h=0.00001):
    """ 
    a naive implementation of numerical gradient of f at x 
    - f should be a function that takes a single argument
    - x is the point (numpy array) to evaluate the gradient at
    """

    fx = f(x)  # evaluate function value at original point
    grad = np.zeros_like(x)
    # iterate over all indexes in x
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:

        # evaluate function at x+h
        ix = it.multi_index
        oldval = x[ix]
        x[ix] = oldval + h  # increment by h
        fxph = f(x)  # evalute f(x + h)
        x[ix] = oldval - h
        fxmh = f(x)  # evaluate f(x - h)
        x[ix] = oldval  # restore

        # compute the partial derivative with centered formula
        grad[ix] = (fxph - fxmh) / (2 * h)  # the slope
        if verbose:
            print(ix, grad[ix])
        it.iternext()  # step to next dimension

    return grad


def eval_numerical_gradient_array(f, x, df, h=1e-5):
    """
    Evaluate a numeric gradient for a function that accepts a numpy
    array and returns a numpy array.
    """
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index

        oldval = x[ix]
        x[ix] = oldval + h
        pos = f(x).copy()
        x[ix] = oldval - h
        neg = f(x).copy()
        x[ix] = oldval

        grad[ix] = np.sum((pos - neg) * df) / (2 * h)
        it.iternext()
    return grad


def eval_numerical_gradient_blobs(f, inputs, output, h=1e-5):
    """
    Compute numeric gradients for a function that operates on input
    and output blobs.

    We assume that f accepts several input blobs as arguments, followed by a blob
    into which outputs will be written. For example, f might be called like this:

    f(x, w, out)

    where x and w are input Blobs, and the result of f will be written to out.

    Inputs: 
    - f: function
    - inputs: tuple of input blobs
    - output: output blob
    - h: step size
    """
    numeric_diffs = []
    for input_blob in inputs:
        diff = np.zeros_like(input_blob.diffs)
        it = np.nditer(input_blob.vals, flags=['multi_index'],
                       op_flags=['readwrite'])
        while not it.finished:
            idx = it.multi_index
            orig = input_blob.vals[idx]

            input_blob.vals[idx] = orig + h
            f(*(inputs + (output,)))
            pos = np.copy(output.vals)
            input_blob.vals[idx] = orig - h
            f(*(inputs + (output,)))
            neg = np.copy(output.vals)
            input_blob.vals[idx] = orig

            diff[idx] = np.sum((pos - neg) * output.diffs) / (2.0 * h)

            it.iternext()
        numeric_diffs.append(diff)
    return numeric_diffs


def eval_numerical_gradient_net(net, inputs, output, h=1e-5):
    return eval_numerical_gradient_blobs(lambda *args: net.forward(),
                                         inputs, output, h=h)


def grad_check_sparse(f, x, analytic_grad, num_checks=10, h=1e-5):
    """
    sample a few random elements and only return numerical
    in this dimensions.
    """

    for i in range(num_checks):
        ix = tuple([randrange(m) for m in x.shape])

        oldval = x[ix]
        x[ix] = oldval + h  # increment by h
        fxph = f(x)  # evaluate f(x + h)
        x[ix] = oldval - h  # increment by h
        fxmh = f(x)  # evaluate f(x - h)
        x[ix] = oldval  # reset

        grad_numerical = (fxph - fxmh) / (2 * h)
        grad_analytic = analytic_grad[ix]
        rel_error = abs(grad_numerical - grad_analytic) / \
            (abs(grad_numerical) + abs(grad_analytic))
        print('numerical: %f analytic: %f, relative error: %e' %
              (grad_numerical, grad_analytic, rel_error))


def visualize_grid(Xs, ubound=255.0, padding=1):
    """
    Reshape a 4D tensor of image data to a grid for easy visualization.

    Inputs:
    - Xs: Data of shape (N, H, W, C)
    - ubound: Output grid will have values scaled to the range [0, ubound]
    - padding: The number of blank pixels between elements of the grid
    """
    (N, H, W, C) = Xs.shape
    grid_size = int(ceil(sqrt(N)))
    grid_height = H * grid_size + padding * (grid_size - 1)
    grid_width = W * grid_size + padding * (grid_size - 1)
    grid = np.zeros((grid_height, grid_width, C))
    next_idx = 0
    y0, y1 = 0, H
    for y in range(grid_size):
        x0, x1 = 0, W
        for x in range(grid_size):
            if next_idx < N:
                img = Xs[next_idx]
                low, high = np.min(img), np.max(img)
                grid[y0:y1, x0:x1] = ubound * (img - low) / (high - low)
                # grid[y0:y1, x0:x1] = Xs[next_idx]
                next_idx += 1
            x0 += W + padding
            x1 += W + padding
        y0 += H + padding
        y1 += H + padding
    # grid_max = np.max(grid)
    # grid_min = np.min(grid)
    # grid = ubound * (grid - grid_min) / (grid_max - grid_min)
    return grid


def vis_grid(Xs):
    """ visualize a grid of images """
    (N, H, W, C) = Xs.shape
    A = int(ceil(sqrt(N)))
    G = np.ones((A*H+A, A*W+A, C), Xs.dtype)
    G *= np.min(Xs)
    n = 0
    for y in range(A):
        for x in range(A):
            if n < N:
                G[y*H+y:(y+1)*H+y, x*W+x:(x+1)*W+x, :] = Xs[n, :, :, :]
                n += 1
    # normalize to [0,1]
    maxg = G.max()
    ming = G.min()
    G = (G - ming)/(maxg-ming)
    return G


def vis_nn(rows):
    """ visualize array of arrays of images """
    N = len(rows)
    D = len(rows[0])
    H, W, C = rows[0][0].shape
    Xs = rows[0][0]
    G = np.ones((N*H+N, D*W+D, C), Xs.dtype)
    for y in range(N):
        for x in range(D):
            G[y*H+y:(y+1)*H+y, x*W+x:(x+1)*W+x, :] = rows[y][x]
    # normalize to [0,1]
    maxg = G.max()
    ming = G.min()
    G = (G - ming)/(maxg-ming)
    return G
