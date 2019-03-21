import tensorflow as tf

def get_init_conv():
    return {
        "w": tf.contrib.layers.xavier_initializer_conv2d(),
        "b": tf.truncated_normal_initializer(stddev=1.0)
    }

def get_init_mlp():
    return {
        "w": tf.contrib.layers.xavier_initializer(),
        "b": tf.truncated_normal_initializer(stddev=1.0)
    }