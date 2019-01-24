import numpy as np
import matplotlib.pyplot as plt


def showimages(X_train, y_train, sample_per_class=8):
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
