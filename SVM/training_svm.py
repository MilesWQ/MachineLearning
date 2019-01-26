import math
from util import plt, np, load_data, grad_check_sparse, time_elapse
from linear_svm import svm_loss_naive, svm_loss_vectorized
from linear_classifier import LinearSVM

cifar_dir = '../cifar-10-batches-py'
X_train, y_train, X_val, y_val, X_test, y_test, X_dev, y_dev = load_data(
    cifar_dir, num_test=500)

# generate a random SVM weight matrix
W = np.random.randn(3073, 10) * 0.0001
"""
gradient check
"""
# loss, grad = svm_loss_naive(W, X_dev, y_dev, 0.0)
# def f(w): return svm_loss_naive(w, X_dev, y_dev, 0.0)[0]
# grad_check_sparse(f, W, grad)

# loss, grad = svm_loss_naive(W, X_dev, y_dev, 1e2)
# def g(w): return svm_loss_naive(w, X_dev, y_dev, 1e2)[0]
# grad_check_sparse(g, W, grad)

# loss, grad = svm_loss_vectorized(W, X_dev, y_dev, 1e2)
# def f(w): return svm_loss_vectorized(w, X_dev, y_dev, 1e2)[0]
# grad_check_sparse(f, W, grad)

"""
test time cost
"""
# print("naive time consume: %f seconds" %time_elapse(svm_loss_naive, W, X_dev, y_dev, 0.0001))
# print("vectorized time consume: %f seconds" %time_elapse(svm_loss_vectorized, W, X_dev, y_dev, 0.0001))

"""
test lost and gradient
"""
# loss_naive, grad_naive = svm_loss_naive(W, X_dev, y_dev, 1e2)
# loss_vectorized, grad_vectorized = svm_loss_vectorized(W, X_dev, y_dev, 1e2)
# loss_difference = (loss_naive - loss_vectorized)
# grad_difference = np.linalg.norm(grad_naive - grad_vectorized, ord='fro')
# print('loss difference: %f' % loss_difference)
# print('gradient difference: %f' % grad_difference)


"""
#test train
svm = LinearSVM()
loss_histtory = svm.train(X_train, y_train, learning_rate=1e-7, reg=5e4, num_iters=1500, verbose=True)
# plot the loss curve
plt.plot(loss_histtory)
plt.xlabel('Iteration number')
plt.ylabel('Loss value')
plt.show()
"""

# tunning hyperparameters
learning_rates = [1e-7, 3e-7, 1e-6, 3e-6, 1e-5, 3e-5,
                  1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1]
regularization_strengths = [1, 3, 1e1, 3e1, 1e2, 3e2, 1e3, 3e3]

# results is dictionary mapping tuples of the form
# (learning_rate, regularization_strength) to tuples of the form
# (training_accuracy, validation_accuracy). The accuracy is simply the fraction
# of data points that are correctly classified.
results = {}
# The highest validation accuracy that we have seen so far.
best_val = -1
# The LinearSVM object that achieved the highest validation rate.
best_svm = None

# lr = learning rate , reg = regularization_strength
for lr in learning_rates:
    for reg in regularization_strengths:
        # new a svm
        svm = LinearSVM()
        # train with training set
        svm.train(X_train, y_train, learning_rate=lr,
                  reg=reg, num_iters=200)
        # get training set accuracy
        y_train_pred = svm.predict(X_train)
        training_accuracy = np.mean(y_train_pred == y_train)
        #print('Training set accuracy is %f' % (training_accuracy,))
        # get validation set accuracy
        y_val_pred = svm.predict(X_val)
        validation_accuracy = np.mean(y_val_pred == y_val)
        #print('Validation set accuracy is %f' % (validation_accuracy,))
        # store the results
        results[(lr, reg)] = (training_accuracy, validation_accuracy)
        if validation_accuracy > best_val:
            best_val = validation_accuracy
            best_svm = svm

for lr, reg in sorted(results):
    train_accuracy, val_accuracy = results[(lr, reg)]
    print('lr %e reg %e train accuracy: %f val accuracy: %f' % (
        lr, reg, train_accuracy, val_accuracy))

print('best validation accuracy achieved during cross-validation: %f' % best_val)

# Visualize the cross-validation results
x_scatter = [math.log10(x[0]) for x in results]
y_scatter = [math.log10(x[1]) for x in results]

# plot training accuracy
marker_size = 100
colors = [results[x][0] for x in results]
plt.subplot(2, 1, 1)
plt.scatter(x_scatter, y_scatter, marker_size, c=colors)
plt.colorbar()
plt.xlabel('log learning rate')
plt.ylabel('log regularization strength')
plt.title('CIFAR-10 training accuracy')

# plot validation accuracy
colors = [results[x][1] for x in results]  # default size of markers is 20
plt.subplot(2, 1, 2)
plt.scatter(x_scatter, y_scatter, marker_size, c=colors)
plt.colorbar()
plt.xlabel('log learning rate')
plt.ylabel('log regularization strength')
plt.title('CIFAR-10 validation accuracy')
plt.show()
