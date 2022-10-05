import tensorflow as tf



@tf.function
def connect_losses(losses:list, trainable=False):
    return tf.add_n([loss * tf.Variable(initial_value=1.0, trainable=trainable) for loss in losses])

