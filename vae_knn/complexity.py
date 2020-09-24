import numpy as np
from sklearn.neighbors import NearestNeighbors
import tensorflow as tf
from tqdm import trange
from vae_models import VAE, vae_loss


def interpolate(model, vae, sample_data, nbrs_data, n=100):
    curr_sum = 0.0
    for i in trange(1, n + 1):
        gamma = (i - 1) / n
        # shape: (interpol_batch_size x latent_dim)
        r1 = (1 - gamma) * sample_data + gamma * nbrs_data
        gamma = i / n
        # shape: (interpol_batch_size x latent_dim)
        r2 = (1 - gamma) * sample_data + gamma * nbrs_data
        o1 = model(vae.decode(r1))
        o2 = model(vae.decode(r2))
        curr_sum += np.linalg.norm(o1 - o2, axis=1)
    return curr_sum


def train(dataset, ds_size):
    batch_size = 256
    latent_dim = 30  # 32, 64, 128 or 10,20,30
    epochs = 20

    ds = dataset.shuffle(ds_size).batch(batch_size)
    vae = VAE(in_channels=3, latent_dim=latent_dim)
    optimizer = tf.keras.optimizers.Adam()

    for epoch in trange(epochs):
        for inputs, targets in ds:
            kld_weight = inputs.shape[0] / ds_size
            with tf.GradientTape() as tape:
                recons, mu, log_var = vae(inputs)
                loss = vae_loss(inputs, recons, mu, log_var, kld_weight)
            gradients = tape.gradient(loss, vae.trainable_weights)
            optimizer.apply_gradients(zip(gradients, vae.trainable_weights))
    return vae


def complexity(model, dataset):
    def num_samples(ds):
        """
        Calculates the number of samples in the given dataset in a slightly more space
        efficient routine than normal.

        :param ds:
        :return:
        """
        return len(list(ds.map(lambda x, y: True)))
    # Hyperparameters
    ds_size = num_samples(dataset)
    n_neighbors = 3
    interpol_batch_size = 64
    avg = tf.keras.metrics.Mean()

    print('+============measure_start============+')
    # Train Variational Autoencoder
    print('|                                     |')
    print('|            Train VAE                |')
    print('|                                     |')
    vae = VAE(in_channels=3, latent_dim=20)  # train(dataset, ds_size) #TODO

    # Train NearestNeighbors
    print('|                                     |')
    print('|            Train KNN                |')
    print('|                                     |')
    # Train Variational Autoencoder
    knn_ds = dataset.shuffle(ds_size)
    # shape: (ds_size x 32 x 32 x 3), (ds_size, )
    knn_train_data, _ = next(iter(knn_ds.batch(ds_size)))
    # shape: (ds_size x latent_dim)
    knn_train_data = vae.get_latent_features(knn_train_data).numpy()
    knn_model = NearestNeighbors(n_neighbors=n_neighbors)
    knn_model.fit(knn_train_data)
    print('|                                     |')
    print('|            Interpolation            |')
    print('|                                     |')

    for step, (x, y) in enumerate(knn_ds.batch(interpol_batch_size)):
        print('|            step:{}                  |'.format(step))
        measure_val = np.zeros(interpol_batch_size)
        latent_samples = vae.get_latent_features(x).numpy()
        nbrs_inds = knn_model.kneighbors(latent_samples, return_distance=False)

        for idx in trange(n_neighbors):
            nbrs_batch = knn_train_data[nbrs_inds[:, idx]]
            measure_val += interpolate(model, vae, latent_samples, nbrs_batch, n=100)
        avg.update_state(measure_val)
        if step == 2:
            break
    print('|                                     |')
    print('+============ measure_end ============+')
    return avg.result().numpy()


def _test():
    from utils import load_model, load_dataset
    # Load Dataset
    dataset = load_dataset()
    # Load Model
    model = load_model()

    result = complexity(model, dataset)
    print(result)


if __name__ == '__main__':
    _test()


