import numpy as np
import tensorflow as tf


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
    accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')
    dataset = dataset.shuffle(ds_size).batch(64)
    for step, (x, y) in enumerate(dataset):
        logits = model(x, training=False)
        accuracy(y, logits)
    return accuracy.result().numpy() * 100.
