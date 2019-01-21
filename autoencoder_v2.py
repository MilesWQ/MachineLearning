import numpy as np
from keras.layers import Input, Dense
from keras.models import Model, Sequential
from keras.datasets import mnist
import matplotlib.pyplot as plt

# load data
(X_train, _), (X_test, _) = mnist.load_data()

# get the input size
input_size = X_train.shape[1] * X_train.shape[2]

# normalize the data
X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255
# flatten data to vectors. the training dataset becomes a 60000x784 matrix and the test dataset becomes 10000x784
X_train = X_train.reshape((len(X_train), np.prod(X_train.shape[1:])))
X_test = X_test.reshape((len(X_test), np.prod(X_test.shape[1:])))

# set the input tenser
inputs = Input(shape=(input_size,))
# set size of encoding
encoding_dim = 64

# encoded tensor
encoded = Dense(encoding_dim, activation='relu')(inputs)
# decoded tensor
decoded = Dense(input_size, activation='sigmoid')(encoded)

# build autoencoder model, maps an input to its reconstruction
autoencoder = Model(inputs, decoded)

# build encoder model, maps an input to its encode representation
encoder = Model(inputs, encoded)

# define a placeholder for an encoded input
encoded_input = Input(shape=(encoding_dim,))

# retrieve the last layer of the autoencoder model
decoder_layer = autoencoder.layers[-1]

# build decoder model
decoder = Model(encoded_input, decoder_layer(encoded_input))

# compile the autoencoder model
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

# train, use the same input as y, epochs =50
autoencoder.fit(X_train, X_train, epochs=50, batch_size=256,
                validation_data=(X_test, X_test))

encoded_images = encoder.predict(X_test)
decoded_images = decoder.predict(encoded_images)
# display images
n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(3, n, i + 1)
    plt.imshow(X_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display encoded
    ax = plt.subplot(3, n, i + 1 + n)
    plt.imshow(encoded_images[i].reshape(8, 8))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(3, n, i + 1 + 2*n)
    plt.imshow(decoded_images[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
