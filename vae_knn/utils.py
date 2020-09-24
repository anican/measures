import os
import tensorflow as tf
import json
import glob


class Sequential(tf.keras.Sequential):
    def __call__(self, x, tape=False, *args, **kwargs):
        if tape:
            tape.watch(x)
        return super(Sequential, self).__call__(x, *args, **kwargs)


def wrap_layer(layer_cls, *args, **kwargs):
    """Wraps a layer for computing the jacobian wrt to intermediate layers."""
    class wrapped_layer(layer_cls):
        def __call__(self, x, *args, **kwargs):
            self._last_seen_input = x
            return super(wrapped_layer, self).__call__(x, *args, **kwargs)
    return wrapped_layer(*args, **kwargs)


def load_dataset():
    task1_path = '/Users/anican/shared/pgdl/public_data/input_data/task1_v4/dataset_1'
    path_to_shards = glob.glob(os.path.join(task1_path, 'train', 'shard_*.tfrecord'))
    print(path_to_shards)
    dataset = tf.data.TFRecordDataset(path_to_shards)
    def _deserialize_example(serialized_example):
        record = tf.io.parse_single_example(
            serialized_example,
            features={
                'inputs': tf.io.FixedLenFeature([], tf.string),
                'output': tf.io.FixedLenFeature([], tf.string)
            })
        inputs = tf.io.parse_tensor(record['inputs'], out_type=tf.float32)
        output = tf.io.parse_tensor(record['output'], out_type=tf.int32)
        return inputs, output
    dataset = dataset.map(_deserialize_example)
    return dataset


def load_model():
    """Loads the model weight and the initial weight, if any."""
    model_directory = '/Users/anican/shared/pgdl/public_data/input_data/task1_v4/model_20'
    with open(os.path.join(model_directory, 'config.json'), 'r') as f:
        config = json.load(f)
    model_instance = model_def_to_keras_sequential(config['model_config'])
    model_instance.build([0] + config['input_shape'])
    weights_path = os.path.join(model_directory, 'weights.hdf5')
    initial_weights_path = os.path.join(model_directory, 'weights_init.hdf5')
    if os.path.exists(initial_weights_path):
        try:
            model_instance.load_weights(initial_weights_path)
            model_instance.initial_weights = model_instance.get_weights()
        except ValueError as e:
            print('Error while loading initial weights of {} from {}'.format('model_20', initial_weights_path))
            print(e)
    model_instance.load_weights(weights_path)
    return model_instance


def model_def_to_keras_sequential(model_def):
    """Convert a model json to a Keras Sequential model.

    Args:
        model_def: A list of dictionaries, where each dict describes a layer to add
            to the model.

    Returns:
        A Keras Sequential model with the required architecture.
    """

    def _cast_to_integer_if_possible(dct):
        dct = dict(dct)
        for k, v in dct.items():
            if isinstance(v, float) and v.is_integer():
                dct[k] = int(v)
        return dct

    def parse_layer(layer_def):
        layer_cls = getattr(tf.keras.layers, layer_def['layer_name'])
        # layer_cls = wrap_layer(layer_cls)
        kwargs = dict(layer_def)
        del kwargs['layer_name']
        return wrap_layer(layer_cls, **_cast_to_integer_if_possible(kwargs))
        # return layer_cls(**_cast_to_integer_if_possible(kwargs))

    return Sequential([parse_layer(l) for l in model_def])
