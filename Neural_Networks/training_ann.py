from util import plt, np, load_data, visualize_grid
from neural_net import TwoLayerNet

cifar_dir = '../cifar-10-batches-py'
X_train, y_train, X_val, y_val, X_test, y_test, X_dev, y_dev = load_data(
    cifar_dir, num_test=500)

input_size = 32 * 32*3
hidden_size = 200
num_classes = 10
num_iters = 3200
best_val = -1
best_nn = None
learning_rates = [1e-3]
regularization_strengths = [5e-6]
learning_history = {}
for lr in learning_rates:
    for reg in regularization_strengths:
        nn = TwoLayerNet(input_size, hidden_size, num_classes)
        report = nn.train(X_train, y_train, X_val, y_val, num_iters=num_iters, batch_size=200,
                          learning_rate=lr, learning_rate_decay=0.98, reg=reg)
        train_acc = np.mean(nn.predict(X_train) == y_train)
        val_acc = np.mean(nn.predict(X_val) == y_val)
        learning_history[(lr, reg)] = (train_acc, val_acc)
        if val_acc > best_val:
            best_val = val_acc
            best_nn = nn

for lr, reg in sorted(learning_history):
    train_acc, val_acc = learning_history[(lr, reg)]
    print('lr %e reg %e train accuracy: %f val accuracy: %f' %
          (lr, reg, train_acc, val_acc))

# print loss curve
plt.subplot(2, 1, 1)
plt.plot(report['loss_history'])
plt.title('Loss history')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.subplot(2, 1, 2)
plt.plot(report['train_acc_history'], label='train')
plt.plot(report['val_acc_history'], label='val')
plt.title('Classification accuracy history')
plt.xlabel('Epoch')
plt.ylabel('Clasification accuracy')
plt.legend()
plt.show()


def show_net_weights(nn):
    # Visualize the weights of the network
    W1 = nn.params['W1']
    W1 = W1.reshape(32, 32, 3, -1).transpose(3, 0, 1, 2)
    plt.imshow(visualize_grid(W1, padding=3).astype('uint8'))
    plt.gca().axis('off')
    plt.show()


show_net_weights(nn)

test_acc = (best_nn.predict(X_test) == y_test).mean()
print('Test accuracy: %f' % (test_acc))
