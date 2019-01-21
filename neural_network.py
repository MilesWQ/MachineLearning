# feedforward propagation neural network
# 0 - 9 number recognition
#  25 units in the second layer(hidden layer) with 10 units in the output layer
import act_func as activation
import numpy as np
import scipy.io as sio

data = sio.loadmat('data.mat')
X = data['X']
y = data['y']
print(y)
theta = sio.loadmat('weights.mat')
theta1 = theta['Theta1']
theta2 = theta['Theta2']


def predict(theta1, theta2, x):
    number_of_input = np.shape(x)
    prediction = np.zeros(number_of_input[0])
    x = np.insert(x, 0, 1, axis=1)  # add bias inputs
    for i in range(number_of_input[0]):
        # compute linear combinations for 2nd layer z = theta * x
        z2 = theta1.dot(x[i])
        # output 2nd layer vector with activation
        a2 = activation.sigmod(z2)
        # add bias into a2
        a2 = np.insert(a2, 0, 1)
        # compute linear combinations for output
        z3 = theta2.dot(a2)
        # compute hypothesis with activation
        hypothesis = activation.sigmod(z3)
        prediction[i] = np.argmax(hypothesis) + 1
    return prediction
