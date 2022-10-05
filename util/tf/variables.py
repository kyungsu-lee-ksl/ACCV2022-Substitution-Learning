import tensorflow as tf


def initPlaceHolders(init_values):
    return tuple([tf.Variable(v, trainable=False) for v in init_values])


def initInputs(input_shape, num=3):
    return tuple([tf.keras.layers.Input(shape=input_shape) for _ in range(0, 3)])
