from abc import abstractmethod
import numpy as np

import tensorflow as tf
from keras.layers import Conv2D, Conv2DTranspose
from tensorflow import keras

from util.tf.layers import DenseBlock


class BaseModel(tf.keras.models.Model):
    sharedDenseBlock01 = DenseBlock(64)
    sharedDenseBlock02 = DenseBlock(128)
    sharedDenseBlock03 = DenseBlock(256)
    sharedDenseBlock04 = DenseBlock(512)

    def __init__(self):
        super().__init__()

    def __conv__(self, output_channel, filter_size=3, activation='relu'):
        return Conv2D(output_channel, (filter_size, filter_size), activation=activation, padding="SAME")

    def __deconv__(self, output_channel, kernel_size=3, activation='relu'):
        return Conv2DTranspose(filters=output_channel, kernel_size=(kernel_size, kernel_size), strides=(2, 2), activation=activation, data_format="channels_last", padding="SAME")

    @abstractmethod
    def call(self, inputs):
        pass

    def build_graph(self, input_shape):
        x = keras.Input(shape=input_shape[1:])
        return keras.Model(inputs=[x], outputs=self.call(x))
