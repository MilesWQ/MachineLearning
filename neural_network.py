import act_func as activation
import numpy as np


def predict(*theta, a1):
    m = np.shape(a1)
    print(m)


X = np.array([[5, 6, 7], [8, 9, 10]])
predict(np.zeros((1, 5)), a1=X)
