import numpy as np
from random import shuffle
"""
copy from cs231n
"""


def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        # a counter record classes that exceed the margin class with (w_jx - w_ix +1 >0)
        class_counter = 0
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1  # note delta = 1
            if margin > 0:
                loss += margin
                class_counter += 1
                # update gradient of xi that isn't the correct class
                dW[:, j] += X[i]
            else:
                dW[:, j] += 0
        # update the the gradient of the correct class of xi
        dW[:, y[i]] += -class_counter * X[i]
    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    # get the average gradient
    dW /= num_train
    # Add regularization to the loss.
    loss += 0.5 * reg * np.sum(W * W)
    # regularized gradient
    dW += reg * W

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################

    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape)  # initialize the gradient as zero
    num_X = X.shape[0]
    delta = 1.0
    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # scores shape N x C
    scores = X.dot(W)
    # correct_class_score shape(N,)
    correct_class_score = scores[np.arange(num_X), y]
    # a margin matrix shape N x C
    margins = np.maximum(
        0, scores - correct_class_score[np.arange(num_X), np.newaxis] + delta)
    # set 0 margin for the correct classes
    margins[np.arange(num_X), y] = 0
    # get mean data loss
    loss = np.sum(margins) / num_X
    # get regularize loss
    loss += 0.5 * reg * np.sum(W*W)
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    """
    a coefficient matrix of dw 0, 1, number counted by margins
    each row of mask is a coefficient set of x_i depending on margins, each number in the  set is a coefficient of one class
    dw_correct_class = - margin_count * x_i
    dw_other_class = (1 or 0) * x_i
    e.g. a mask is [1, 0, margin_count]
    """
    mask = np.zeros(margins.shape)
    mask[margins > 0] = 1
    # a column of counted number of margin
    margin_counter = np.sum(mask, axis=1)
    # update the coefficient of the correct class
    mask[np.arange(num_X), y] = -margin_counter
    # compute total dW
    # X = examples x weight X.T = weight x examples shape DxN , mask shape NxC
    dW = X.T.dot(mask)
    # get the mean dW
    dW /= num_X
    # regularized Dw
    dW += reg * W
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return loss, dW
