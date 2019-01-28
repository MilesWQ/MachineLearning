import numpy as np
from random import shuffle


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

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
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    pass
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

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
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    num_X = X.shape[0]
    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # scores shape = [NxD].[DxC] = [NXC]
    scores = X.dot(W)
    c = np.max(scores, axis=1)
    # apply numeric stability
    scores -= c[np.arange(num_X), np.newaxis]
    # exps shape = [NxC]
    exps = np.exp(scores)
    # normalize
    exp_sum = np.sum(exps, axis=1, keepdims=True)
    # shape [NXC]
    exps /= exp_sum
    # get the correct class exps
    correct_classes = exps[np.arange(num_X), y]
    # regularized mean loss
    loss = np.sum(-np.log(correct_classes)) / num_X + 0.5 * reg * np.sum(W*W)

    # compute gradient
    correct_classes_mask = np.zeros(exps.shape)
    correct_classes_mask[np.arange(num_X), y] = 1
    # NXC with the correct class s_j-1
    soft_max_gradients = exps - correct_classes_mask
    dW = X.T.dot(soft_max_gradients) / num_X + reg * W
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW
