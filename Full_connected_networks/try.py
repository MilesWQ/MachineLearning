import numpy as np

h = [np.array([1, 2, 3]), np.array([7, 8, 9])]
h.reverse()
for i, h in enumerate(h):
    if i == 0:
        print('111')
    print(h)
