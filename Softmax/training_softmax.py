from util import plt, np, load_data, grad_check_sparse, time_elapse
from softmax import softmax_loss_vectorized
from linear_classifier import Softmax

cifar_dir = '../cifar-10-batches-py'
X_train, y_train, X_val, y_val, X_test, y_test, X_dev, y_dev = load_data(
    cifar_dir, num_test=500)

# ininialize W
W = np.random.randn(3073, 10) * 0.0001

# test loss
loss, grad = softmax_loss_vectorized(W, X_dev, y_dev, 0.0)
#print('loss: %f' % loss)
#print('sanity check: %f' % (-np.log(0.1)))

# test gradient without regularization
#def f(w): return softmax_loss_vectorized(W, X_dev, y_dev, 0.0)[0]
#grad_numerical = grad_check_sparse(f, W, grad, 10)

# test gradient with regularization
#def f(w): return softmax_loss_vectorized(W, X_dev, y_dev, 1e2)[0]
#grad_numerical = grad_check_sparse(f, W, grad, 10)

softmax = Softmax()
loss_history = softmax.train(
    X_train, y_train, learning_rate=1e-7, reg=5e4, num_iters=1500, verbose=True)
plt.plot(loss_history)
plt.xlabel('Iteration number')
plt.ylabel('Loss value')
plt.show()
