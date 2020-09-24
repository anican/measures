import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


class AE1(Model):
    def __init__(self, latent_dim):
        super(AE1, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = tf.keras.Sequential([
            layers.Input(shape=(32, 32, 3)),
            # layers.Conv2D(64, (3, 3), activation='relu', padding='same', strides=2),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same', strides=2),
            layers.Conv2D(16, (3, 3), activation='relu', padding='same', strides=2),
            # layers.Conv2D(8, (3, 3), activation='relu', padding='same', strides=2),
            layers.Flatten(),
            layers.Dense(latent_dim)
        ])

        self.decoder = tf.keras.Sequential([
            layers.Dense(1024),
            layers.Reshape((8, 8, 16)),
            # layers.Conv2DTranspose(8, kernel_size=3, strides=2, activation='relu',
            #                        padding='same'),
            layers.Conv2DTranspose(16, kernel_size=3, strides=2, activation='relu',
                                   padding='same'),
            layers.Conv2DTranspose(32, kernel_size=3, strides=2, activation='relu',
                                   padding='same'),
            # layers.Conv2DTranspose(64, kernel_size=3, strides=2, activation='relu',
            #                        padding='same'),
            layers.Conv2D(3, kernel_size=(3, 3), activation='sigmoid', padding='same')
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class AE2(Model):
    def __init__(self, latent_dim):
        super(AE2, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = tf.keras.Sequential([
            layers.Input(shape=(32, 32, 3)),
            # layers.Conv2D(64, (3, 3), activation='relu', padding='same', strides=2),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same', strides=2),
            layers.Conv2D(16, (3, 3), activation='relu', padding='same', strides=2),
            layers.Conv2D(8, (3, 3), activation='relu', padding='same', strides=2),
            layers.Flatten(),
            layers.Dense(latent_dim)
        ])

        self.decoder = tf.keras.Sequential([
            layers.Dense(128),
            layers.Reshape((4, 4, 8)),
            layers.Conv2DTranspose(8, kernel_size=3, strides=2, activation='relu',
                                   padding='same'),
            layers.Conv2DTranspose(16, kernel_size=3, strides=2, activation='relu',
                                   padding='same'),
            layers.Conv2DTranspose(32, kernel_size=3, strides=2, activation='relu',
                                   padding='same'),
            # layers.Conv2DTranspose(64, kernel_size=3, strides=2, activation='relu',
            #                        padding='same'),
            layers.Conv2D(3, kernel_size=(3, 3), activation='sigmoid', padding='same')
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class AE3(Model):
    def __init__(self, latent_dim):
        super(AE3, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = tf.keras.Sequential([
            layers.Input(shape=(32, 32, 3)),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same', strides=2),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same', strides=2),
            layers.Conv2D(16, (3, 3), activation='relu', padding='same', strides=2),
            # layers.Conv2D(8, (3, 3), activation='relu', padding='same', strides=2),
            layers.Flatten(),
            layers.Dense(latent_dim)
        ])

        self.decoder = tf.keras.Sequential([
            layers.Dense(256),
            layers.Reshape((4, 4, 16)),
            # layers.Conv2DTranspose(8, kernel_size=3, strides=2, activation='relu',
            #                        padding='same'),
            layers.Conv2DTranspose(16, kernel_size=3, strides=2, activation='relu',
                                   padding='same'),
            layers.Conv2DTranspose(32, kernel_size=3, strides=2, activation='relu',
                                   padding='same'),
            layers.Conv2DTranspose(64, kernel_size=3, strides=2, activation='relu',
                                   padding='same'),
            layers.Conv2D(3, kernel_size=(3, 3), activation='sigmoid', padding='same')
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def test_ae1():
    x = np.random.randn(200, 32, 32, 3)
    orig_shape = x.shape
    model = AE1(10)
    y = model(x)
    assert y.shape == orig_shape
    model = AE1(20)
    y = model(x)
    assert y.shape == orig_shape
    model = AE1(30)
    y = model(x)
    assert y.shape == orig_shape
    y = model(x)
    assert y.shape == orig_shape
    print('AE1 tests finished')


def test_ae2():
    x = np.random.randn(200, 32, 32, 3)
    orig_shape = x.shape
    model = AE2(10)
    y = model(x)
    assert y.shape == orig_shape
    model = AE2(20)
    y = model(x)
    assert y.shape == orig_shape
    model = AE2(30)
    y = model(x)
    assert y.shape == orig_shape
    y = model(x)
    assert y.shape == orig_shape
    print('AE2 tests finished')



def test_ae3():
    x = np.random.randn(200, 32, 32, 3)
    orig_shape = x.shape
    model = AE3(10)
    y = model(x)
    assert y.shape == orig_shape
    model = AE3(20)
    y = model(x)
    assert y.shape == orig_shape
    model = AE3(30)
    y = model(x)
    assert y.shape == orig_shape
    y = model(x)
    assert y.shape == orig_shape
    print('AE3 tests finished')


if __name__ == '__main__':
    model = AE2(30)
    test_ae1()
    test_ae2()
    test_ae3()
