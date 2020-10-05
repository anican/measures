import numpy as np
import tensorflow as tf


def complexity(model, ds):
    def get_size(ds):
        return len(list(ds.map(lambda x, y: True)))
    n = 100
    batch_size = 64
    ds_size = get_size(ds)
    ds = ds.shuffle(ds_size).batch(batch_size * 2)
    avg = tf.keras.metrics.Mean()
    for step, (x, y) in enumerate(ds):
        measure_val = 0.
        r1 = x[:batch_size]
        r2 = x[batch_size:]
        for i in range(1, n+1):
            gamma = (i - 1) / n
            rhat1 = (1 - gamma) * r1 + gamma * r2
            gamma = i / n
            rhat2 = (1 - gamma) * r1 + gamma * r2
            logits_delta = model(rhat1) - model(rhat2)
            measure_val += np.linalg.norm(logits_delta, axis=1)
        avg.update_state(measure_val)
        if step == 7:
            break
    return avg.result().numpy()


