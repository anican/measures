import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D
from  tensorflow.keras import Sequential
from tensorflow.keras import backend as K


tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


class VAE(tf.keras.Model):
    def __init__(self, in_channels: int, latent_dim: int):
        super(VAE, self).__init__()
        self.in_channels = in_channels

        # encoder modules
        self._encoder = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, kernel_size=3, strides=2, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Conv2D(64, kernel_size=3, strides=2, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Conv2D(128, kernel_size=3, strides=2, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Conv2D(256, kernel_size=3, strides=2, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
        ])
        self.fc_mu = tf.keras.layers.Dense(latent_dim, input_shape=(None, 256*2*2))
        self.fc_log_var = tf.keras.layers.Dense(latent_dim, input_shape=(None, 256*2*2))

        # decoder modules
        self.decoder_input = tf.keras.layers.Dense(256*2*2, input_shape=(None, latent_dim))
        self._decoder = tf.keras.Sequential([
            tf.keras.layers.Conv2DTranspose(256, kernel_size=3, strides=2, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Conv2DTranspose(128, kernel_size=3, strides=2, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Conv2DTranspose(64, kernel_size=3, strides=2, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Conv2DTranspose(32, kernel_size=3, strides=2, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
        ])
        self.final_layer = tf.keras.layers.Conv2D(in_channels, kernel_size=3,
                                                  padding='same', activation='sigmoid')

    def encode(self, inputs):
        result = self._encoder(inputs)
        result = tf.reshape(result, (result.shape[0], -1))
        mu = self.fc_mu(result)
        log_var = self.fc_log_var(result)
        return [mu, log_var]

    def get_latent_features(self, inputs):
        return reparameterize(*self.encode(inputs))

    def decode(self, z):
        result = self.decoder_input(z)
        result = tf.reshape(result, (-1, 2, 2, 256))
        result = self._decoder(result)
        result = self.final_layer(result)
        return result

    def call(self, inputs):
        mu, log_var = self.encode(inputs)
        z = reparameterize(mu, log_var)
        recons = self.decode(z)
        return [recons, mu, log_var]


def reparameterize(mu, log_var):
    std = np.exp(0.5 * log_var)
    eps = np.random.randn(*std.shape)
    return eps * std + mu


def vae_loss(inputs, recons, mu, log_var, kld_weight):
    """

    :param inputs:
    :param recons:
    :param mu:
    :param log_var:
    :param kld_weight: # minibatch samples / total samples
    :return:
    """
    mse_criterion = tf.keras.losses.MeanSquaredError()
    mse_loss = mse_criterion(inputs, recons)
    kld_loss = K.mean(-0.5 * K.sum(1 + log_var - K.square(mu) - K.exp(log_var), axis=1), axis=0)
    loss = mse_loss + kld_weight * kld_loss
    return loss


if __name__ == '__main__':
    model = VAE(3, 10, 20)
    x = np.random.randn(100, 32, 32, 3)
    print()


