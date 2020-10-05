import numpy as np
import os
import tensorflow as tf
from sklearn.neighbors import kneighbors_graph
from vae_models import VAE
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def interpolate(model, vae, data, data_nbrs, n=100):
    curr_sum = 0.0
    for i in range(1, n + 1):
        gamma = (i - 1) / n
        # shape: (interpol_batch_size x latent_dim)
        r1 = (1 - gamma) * data + gamma * data_nbrs
        gamma = i / n
        # shape: (interpol_batch_size x latent_dim)
        r2 = (1 - gamma) * data + gamma * data_nbrs
        o1 = model(vae.decode(r1))
        o2 = model(vae.decode(r2))
        curr_sum += np.linalg.norm(o1 - o2, axis=1)
    return curr_sum


def complexity(model, dataset, program_dir):
    def num_samples(ds):
        """
        Calculates the number of samples in the given dataset in a slightly more space
        efficient routine than normal.

        :param ds:
        :return:
        """
        return len(list(ds.map(lambda x, y: True)))
    ds_size = num_samples(dataset)
    n_neighbors = 3
    batch_size = 64
    avg = tf.keras.metrics.Mean()
    print('+============measure_start============+')
    # Train Variational Autoencoder
    print('|                                     |')
    print('|             Load VAE                |')
    print('|                                     |')
    # TODO: load variational autoencoder
    vae = VAE(in_channels=3, latent_dim=20)
    build_sample = tf.zeros((16, 32, 32, 3))
    build_out = vae(build_sample)
    if ds_size == 50000:
        vae.load_weights(os.path.join(program_dir, 'vae_cifar10.h5'))
    else:
        vae.load_weights(os.path.join(program_dir, 'vae_svhn.h5'))

    # Train NearestNeighbors
    print('|                                     |')
    print('|            Train KNN                |')
    print('|                                     |')
    cutoff = 10000  # only train KNN on first cutoff points, shuffle is critical!
    ds_knn = dataset.shuffle(ds_size)
    train_data, _ = next(iter(ds_knn.batch(cutoff)))
    train_data = train_data.numpy()
    train_data = vae.get_latent_features(train_data).numpy()
    nbrs_graph = kneighbors_graph(train_data, n_neighbors=n_neighbors, n_jobs=-1,
                                  include_self=False)
    nbrs_inds = np.array(np.split(nbrs_graph.indices, nbrs_graph.indptr)[1:-1])
    n_steps = cutoff // batch_size
    for step in range(n_steps):
        print('|                                     |')
        print('|            step: {}                  |'.format(step))
        measure_val = np.zeros(batch_size)
        # shape: (batch_size x n_neighbors x latent_dim)
        x = train_data[step*batch_size:(step+1)*batch_size]
        nbrs = train_data[nbrs_inds[step*batch_size:(step+1)*batch_size]]
        for idx in range(n_neighbors):
            measure_val += interpolate(model, vae, x, nbrs[:, idx, :], n=100)
        avg.update_state(measure_val)
        if step == 2:
            break
    print('|                                     |')
    print('+============ measure_end ============+')
    return avg.result().numpy()




