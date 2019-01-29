from neural_net import TwoLayerNet, np, plt
from util import load_data, eval_numerical_gradient

# create a small net and toy data
# sea a random seed for repeatable experiments
input_size = 4
hidden_size = 10
num_classes = 3
num_inputs = 5

cifar_dir = '../cifar-10-batches-py'
X_train, y_train, X_val, y_val, X_test, y_test, X_dev, y_dev = load_data(
    cifar_dir, num_dev=10)


def init_toy_model(input_size, hidden_size, num_classes):
    np.random.seed(0)
    return TwoLayerNet(input_size, hidden_size, num_classes, std=1e-1)


def init_toy_data():
    np.random.seed(1)
    # X shape NxD
    X = 10 * np.random.randn(num_inputs, input_size)
    # y shape (N,)
    y = np.array([0, 1, 2, 2, 1])
    return X, y


nn = init_toy_model(input_size, hidden_size, num_classes)
X, y = init_toy_data()

# test scores
"""
scores = nn.loss(X)
print('Your scores:')
print(scores)
print()
print('correct scores:')
correct_scores = np.asarray([
    [-0.81233741, -1.27654624, -0.70335995],
    [-0.17129677, -1.18803311, -0.47310444],
    [-0.51590475, -1.01354314, -0.8504215],
    [-0.15419291, -0.48629638, -0.52901952],
    [-0.00618733, -0.12435261, -0.15226949]])
print(correct_scores)
print('Difference between your scores and correct scores:')
print(np.sum(np.abs(scores - correct_scores)))
"""
# test loss
loss, _ = nn.loss(X, y, reg=0.1)
correct_loss = 1.30378789133
# should be very small, we get < 1e-12
print('Difference between your loss and correct loss:')
print(np.sum(np.abs(loss - correct_loss)))


# Test loss 2
input_size = 32*32*3
hidden_size = 10
num_classes = 10
nn2 = init_toy_model(input_size, hidden_size, num_classes)
loss, _ = nn2.loss(X_train, y_train, reg=0.1)
print('train loss:', loss)
"""
def rel_error(x, y):
    # returns relative error
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))


loss, grads = nn.loss(X, y, 0.1)
for param_name in grads:
    def f(W): return nn.loss(X, y, reg=0.1)[0]
    param_grad_num = eval_numerical_gradient(
        f, nn.params[param_name], verbose=False)
    print('%s max relative error: %e' %
          (param_name, rel_error(param_grad_num, grads[param_name])))

ann = init_toy_model()
stats = ann.train(X, y, X, y,
                  learning_rate=1e-1, reg=1e-5,
                  num_iters=100, verbose=False)

print('Final training loss: ', stats['loss_history'][-1])

# plot the loss history
plt.plot(stats['loss_history'])
plt.xlabel('iteration')
plt.ylabel('training loss')
plt.title('Training Loss history')
plt.show()
"""
