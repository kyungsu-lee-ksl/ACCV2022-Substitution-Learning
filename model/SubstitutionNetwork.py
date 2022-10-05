import keras
import tensorflow as tf
from keras.layers import Conv2D, Flatten, Dense, MaxPooling2D, Conv2DTranspose

from model.BaseModel import BaseModel
from util.tf.layers import DenseBlock


class Architecture(BaseModel):

    def __init__(self, vgg_shape=(512, 512)):
        super().__init__()
        self.conv1 = self.__conv__(64)
        self.conv2 = self.__conv__(128)
        self.conv3 = self.__conv__(256)
        self.conv4 = self.__conv__(64)
        self.conv5 = self.__conv__(1)

        self.vgg_shape = vgg_shape

        self.content_layers = ['block5_conv2']
        self.style_layers = ['block1_conv1',
                             'block2_conv1',
                             'block3_conv1',
                             'block4_conv1',
                             'block5_conv1']

        self.vgg = self.vgg_layers(self.content_layers + self.style_layers)
        self.style_layers = self.style_layers
        self.content_layers = self.content_layers
        self.num_style_layers = len(self.style_layers)
        self.vgg.trainable = False

    def vgg_layers(self, layer_name):
        vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet', input_shape=(*self.vgg_shape, 3))
        vgg.trainable = False
        outputs = [vgg.get_layer(name).output for name in layer_name]
        model = tf.keras.Model([vgg.input], outputs)
        return model

    def gram_matrix(self, input_tensor):
        result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
        input_shape = tf.shape(input_tensor)
        num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)
        return result / num_locations

    def simple_cnn(self, input):
        x1 = self.conv1(input)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3) + x1
        return self.conv5(x4)

    def contextTerm(self, inputs):
        inputs = tf.concat([inputs] * 3, axis=-1)
        preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
        outputs = self.vgg(preprocessed_input)

        style_outputs, content_outputs = (outputs[:self.num_style_layers],
                                          outputs[self.num_style_layers:])

        style_outputs = [self.gram_matrix(style_output)
                         for style_output in style_outputs]

        content_dict = {content_name: value
                        for content_name, value in zip(self.content_layers, content_outputs)}
        style_dict = {style_name: value
                      for style_name, value in zip(self.style_layers, style_outputs)}

        return {'content': content_dict, 'style': style_dict}


    @tf.function
    def __loss__(self, req):
        return None



    def call(self, inputs):
        mri_input, tdn_input, rct_input = inputs

        tdn_mask = tf.cast(mri_input, tf.float32) * tf.cast(tdn_input > 0, tf.float32)
        rct_mask = tf.cast(mri_input, tf.float32) * tf.cast(rct_input > 0, tf.float32)

        tdn_fft = tf.signal.fft2d(tf.complex(tdn_mask, tf.zeros_like(tdn_mask)))
        rct_fft = tf.signal.fft2d(tf.complex(rct_mask, tf.zeros_like(rct_mask)))

        tdnReal = tf.math.real(tdn_fft)
        tdnImag = tf.math.imag(tdn_fft)

        rctReal = self.simple_cnn(tf.math.real(rct_fft))
        rctImag = self.simple_cnn(tf.math.imag(rct_fft))

        outputs = {
            # substituted image
            'subst': tf.math.real(tf.signal.ifft2d(tf.complex(rctReal, rctImag))),
            # content & style loss
            'tdn': {
                'real': self.contextTerm(tdnReal),
                'imag': self.contextTerm(tdnImag)
            },
            'rct': {
                'real': self.contextTerm(rctReal),
                'imag': self.contextTerm(rctImag)
            },
        }
        # outputs['loss'] = self.__loss__(outputs)

        return outputs
