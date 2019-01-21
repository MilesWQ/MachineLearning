import numpy as np
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import backend as K
from keras.datasets import mnist
import matplotlib.pyplot as plt

'''
reference from https://blog.keras.io/building-autoencoders-in-keras.html
'''

# load data
(X_train, _), (X_test, _) = mnist.load_data()

# get the shape of the image
input_shape = (X_train.shape[1], X_train.shape[2], 1)

# normalize the data
X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255

# reshape data to 28x28x1
X_train = X_train.reshape((len(X_train), 28, 28, 1))
X_test = X_test.reshape((len(X_test), 28, 28, 1))

# set input tensor, reserve the original shape
inputs = Input(shape=input_shape)

# 1st conv with 16 3x3 filter, output spatial is the same as input 28x28
layers = Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
# 1st pooling with 2x2 pool size, stripe size is default to the pool size (2) output
layers = MaxPooling2D((2, 2), padding='same')(layers)
# 2nd conv with 8 3x3 filter output 14X14
layers = Conv2D(8, (3, 3), activation='relu', padding='same')(layers)
# 2nd pooling with 2x2 pool size
layers = MaxPooling2D((2, 2), padding='same')(layers)
# 3rd conv with 8 3x3 filter
layers = Conv2D(8, (3, 3), activation='relu', padding='same')(layers)
# output pooling as the encoded tensor
encoded = MaxPooling2D((2, 2), padding='same')(layers)

# construct decoder
layers = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
layers = UpSampling2D((2, 2))(layers)
layers = Conv2D(8, (3, 3), activation='relu', padding='same')(layers)
layers = UpSampling2D((2, 2))(layers)
# no padding
layers = Conv2D(16, (3, 3), activation='relu')(layers)
layers = UpSampling2D((2, 2))(layers)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(layers)

# construct autoencoder
autoencoder = Model(inputs, decoded)

# print a model summary
autoencoder.summary()

# compile the model
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
# training
autoencoder.fit(X_train, X_train, epochs=50, batch_size=128,
                validation_data=(X_test, X_test))

autoencoder.save('conv_autoencoder.h5')

decoded_images = autoencoder.predict(X_test)

# display images
n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(3, n, i + 1)
    plt.imshow(X_test[i].reshape(28, 28))
    plt.gray()
    plt.title('original')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(3, n, i + 1 + 2*n)
    plt.imshow(decoded_images[i].reshape(28, 28))
    plt.gray()
    plt.title('decoded')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
