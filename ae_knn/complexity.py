# Import statements
import numpy as np
import tensorflow as tf
from sklearn.neighbors import KNeighborsClassifier as KNN
from autoencoder_models import AE1


# import glob
# import os
# import json

# Suppress warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def interpolate(model, ae_model, sample_data, nbrs_data, n=100):
    """

    :param model: neural network classifier model in PGDL format
    :param ae_model: trained autoencoder model
    :param sample_data: samples drawn from a PGDL dataset
    :param nbrs_data: neighbors to the sample_data based on a trained KNN model.
    :param n: interpolation range
    :return:
    """
    curr_sum = 0.0
    for i in range(1, n + 1):
        gamma = (i - 1) / n
        # shape: (interpol_batch_size x latent_dim)
        r1 = (1 - gamma) * sample_data + gamma * nbrs_data
        gamma = i / n
        # shape: (interpol_batch_size x latent_dim)
        r2 = (1 - gamma) * sample_data + gamma * nbrs_data
        o1 = model(ae_model.decoder(r1))
        o2 = model(ae_model.decoder(r2))
        curr_sum += np.linalg.norm(o1 - o2, axis=1)
    return curr_sum


def train_autoencoder(dataset, dataset_size):
    latent_dim = 20
    ae_batch_size = 128
    ae_epochs = 10  # TODO: change to actual value
    # lr = 0.001

    ae_dataset = dataset.shuffle(dataset_size).batch(ae_batch_size)
    model = AE1(latent_dim)
    optimizer = tf.keras.optimizers.Adam()  # TODO: set lr
    criterion = tf.keras.losses.MeanSquaredError()

    for epoch in range(ae_epochs):
        for x, y in ae_dataset:
            with tf.GradientTape() as tape:
                x_recons = model(x)
                loss = criterion(x, x_recons)
            gradients = tape.gradient(loss, model.trainable_weights)
            optimizer.apply_gradients(zip(gradients, model.trainable_weights))
    return model


def complexity(model, dataset):
    def num_samples(ds):
        """
        Calculates the number of samples in the given dataset in a slightly more space
        efficient routine than normal.

        :param ds:
        :return:
        """
        return len(list(ds.map(lambda x, y: True)))

    #    Hyperparameters    #
    dataset_size = num_samples(dataset)
    n_neighbors = 3
    interpol_batch_size = 64
    avg = tf.keras.metrics.Mean()

    #    Train Autoencoder    #
    autoencoder = train_autoencoder(dataset, dataset_size)

    #    Train KNN on latent features    #
    knn_ds = dataset.shuffle(dataset_size)
    # shape: (ds_size x 32 x 32 x 3), (ds_size, )
    knn_train_data, knn_train_targets = next(iter(knn_ds.batch(dataset_size)))
    # shape: (ds_size x latent_dim)
    knn_train_data = autoencoder.encoder(knn_train_data)
    knn_train_data, knn_train_targets = knn_train_data.numpy(), knn_train_targets.numpy()
    knn_model = KNN(n_neighbors=n_neighbors)
    knn_model.fit(knn_train_data, knn_train_targets)

    #    Interpolation    #
    # TODO: fix size and plot measure val avg vs number of samples used in interpolation
    for step, (x, y) in enumerate(knn_ds.batch(interpol_batch_size)):
        measure_val = 0.0
        # shape: (interpol_batch_size x latent_dim)
        samples = autoencoder.encoder(x).numpy()
        # shape: (interpol_batch_size x n_neighbors)
        nbrs_inds = knn_model.kneighbors(samples, return_distance=False)

        for idx in range(n_neighbors):
            nbrs_batch = knn_train_data[nbrs_inds[:, idx]]
            measure_val += interpolate(model, autoencoder, samples, nbrs_batch, n=100)

        avg.update_state(measure_val)
        if step == 2:
            break
    print('model queried...')
    return avg.result().numpy()


# def _test():
#     from utils import load_model, load_dataset
#     # Load Dataset
#     dataset = load_dataset()
#     # Load Model
#     model = load_model()
#
#     result = complexity(model, dataset)
#     print(result)
#
#
# if __name__ == '__main__':
#     _test()
