
# coding: utf-8

# In[7]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import gzip
import csv
import struct
import os
import time
import tensorboard
import keras
from keras import backend as K

from keras.layers import Input, Dense, Lambda, Layer, Add, Multiply
from keras.models import Model, Sequential
from keras.datasets import mnist

from sklearn.model_selection import train_test_split


K.clear_session()
def nll(y_true, y_pred):
    """ Negative log likelihood (Bernoulli). """

    # keras.losses.binary_crossentropy gives the mean
    # over the last axis. we require the sum
    return K.sum(K.binary_crossentropy(y_true, y_pred), axis=-1)


class KLDivergenceLayer(Layer):

    """ Identity transform layer that adds KL divergence
    to the final model loss.
    """

    def __init__(self, *args, **kwargs):
        self.is_placeholder = True
        super(KLDivergenceLayer, self).__init__(*args, **kwargs)

    def call(self, inputs):

        mu, log_var = inputs

        kl_batch = - .5 * K.sum(1 + log_var -
                                K.square(mu) -
                                K.exp(log_var), axis=-1)

        self.add_loss(K.mean(kl_batch), inputs=inputs)

        return inputs
    
    
    
    
try:
    from urllib.request import urlretrieve
except ImportError:
    from urllib import urlretrieve

# Define data loading functions
#   based on code found at
#       https://cntk.ai/pythondocs/CNTK_103A_MNIST_DataLoader.html


def LoadImages(imgurl, cimg):
    print('Loading Images for ' + imgurl)
    gzfname, h = urlretrieve(imgurl, './delete.me')
    with gzip.open(gzfname) as gz:
        struct.unpack('I', gz.read(4))
        struct.unpack('>I', gz.read(4))
        crow = struct.unpack('>I', gz.read(4))[0]
        ccol = struct.unpack('>I', gz.read(4))[0]
        res = np.fromstring(gz.read(cimg * crow * ccol),
                            dtype=np.uint8)
    os.remove(gzfname)
    return res.reshape((cimg, crow * ccol))

def LoadLabels(laburl, cimg):
    print('Loading Labels for ' + laburl)
    gzfname, h = urlretrieve(laburl, './delete.me')
    with gzip.open(gzfname) as gz:
        struct.unpack('I', gz.read(4))
        struct.unpack('>I', gz.read(4))
        
        res = np.fromstring(gz.read(cimg), dtype=np.uint8)
    os.remove(gzfname)
    return res
    # return res.reshape((cimg, 1))


#       Functions for visualizing MNIST images
#       ******* DO WE NEED? USE TENSORBOARD INSTEAD ********
def plotData(recIDX, img_data, lab_data, titleSTR):
    plt.imshow(img_data[recIDX].reshape(28, 28), cmap="gray_r")
    s = titleSTR + ': %i\n' % lab_data[recIDX]
    plt.title(s, fontsize=18)
    plt.axis('off')
    plt.show()

def getModel(intermediate_dim,latent_dim,original_dim,epsilon_std ):

    decoder = Sequential([
        Dense(intermediate_dim, input_dim=latent_dim, activation='relu'),
        Dense(original_dim, activation='sigmoid')
    ])

    x = Input(shape=(original_dim,))
    h = Dense(intermediate_dim, activation='relu')(x)

    z_mu = Dense(latent_dim)(h)
    z_log_var = Dense(latent_dim)(h)

    z_mu, z_log_var = KLDivergenceLayer()([z_mu, z_log_var])
    z_sigma = Lambda(lambda t: K.exp(.5*t))(z_log_var)

    eps = Input(tensor=K.random_normal(stddev=epsilon_std,
                                       shape=(K.shape(x)[0], latent_dim)))
    z_eps = Multiply()([z_sigma, eps])
    z = Add()([z_mu, z_eps])

    x_pred = decoder(z)

    vae = Model(inputs=[x, eps], outputs=x_pred)
    vae.compile(optimizer='rmsprop', loss=nll)
    
    encoder = Model(x, z_mu)
    return vae, encoder, decoder


##############################################################################
def getData(original_dim):
    # URLs for data
    url_train_image = 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz'
    url_train_labels = 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz'
    num_train_samples = 60000

    # get TRAINING IMAGES
    x_train = LoadImages(url_train_image, num_train_samples)

    # get TRAINING LABELS
    y_train = LoadLabels(url_train_labels, num_train_samples)



    # get TEST IMAGES
    url_test_image = 'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz'
    # get TEST LABELS
    url_test_labels = 'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz'
    num_test_samples = 10000

    # get test images
    x_test = LoadImages(url_test_image, num_test_samples)

    # get test labels
    y_test = LoadLabels(url_test_labels, num_test_samples)



    x_train = x_train.reshape(-1, original_dim) / 255.
    x_test = x_test.reshape(-1, original_dim) / 255.

    return x_train, x_test, y_train, y_test


# In[ ]:

logfile = './logs/tb_test'
original_dim = 784
intermediate_dim = 256
batch_size = 100
epochs = 10
epsilon_std = 1.0
x_train_orig, x_test, y_train_orig, y_test = getData(original_dim)

for latent_dim in [2, 10, 20]:
    K.clear_session()

    
    x_train, x_val, y_train, y_val = train_test_split(
                                    x_train_orig, y_train_orig, test_size=0.15)
    vae, encoder, decoder = getModel(intermediate_dim,latent_dim,original_dim,epsilon_std)

    tensorboardcall = [keras.callbacks.TensorBoard(log_dir=logfile, histogram_freq=0, batch_size=batch_size, write_graph=True,
                                write_grads=True, write_images=True, embeddings_freq=0, embeddings_layer_names=None,
                                )]


    vae.fit(x_train,
            x_train,
            shuffle=True,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(x_val, x_val),
            callbacks=tensorboardcall)



    # display a 2D plot of the digit classes in the latent space

    z_test = encoder.predict(x_test, batch_size=batch_size)
    plt.figure(figsize=(6, 6))
    plt.scatter(z_test[:, 0], z_test[:, 1], c=y_test,
                alpha=.4, s=3**2, cmap='viridis')
    plt.colorbar()
    plt.show()

    # display a 2D manifold of the digits
    n = 15  # figure with 15x15 digits
    digit_size = 28

    # linearly spaced coordinates on the unit square were transformed
    # through the inverse CDF (ppf) of the Gaussian to produce values
    # of the latent variables z, since the prior of the latent space
    # is Gaussian
    u_grid = np.dstack(np.meshgrid(np.linspace(0.05, 0.95, n),
                                   np.linspace(0.05, 0.95, n)))
    z_grid = norm.ppf(u_grid)
    x_decoded = decoder.predict(z_grid.reshape(n*n, 2))

    x_decoded = x_decoded.reshape(n, n, digit_size, digit_size)

    plt.figure(figsize=(10, 10))
    plt.imshow(np.block(list(map(list, x_decoded))), cmap='gray')
    plt.show()

