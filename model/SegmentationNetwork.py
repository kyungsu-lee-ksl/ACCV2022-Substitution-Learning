from keras.layers import MaxPooling2D, Concatenate

from model.BaseModel import BaseModel
from util.tf.layers import DenseBlock


class Architecture(BaseModel):

    def __init__(self):
        super().__init__()
        self.denseBlock05 = DenseBlock(1024)
        self.denseBlock06 = DenseBlock(512)
        self.denseBlock07 = DenseBlock(256)
        self.denseBlock08 = DenseBlock(128)
        self.denseBlock09 = DenseBlock(128)

        self.conv1 = self.__conv__(64)
        self.conv2 = self.__conv__(128)
        self.conv3 = self.__conv__(256)
        self.conv4 = self.__conv__(512)
        self.conv5 = self.__conv__(1024)
        self.conv6 = self.__conv__(3, activation='softmax')

        self.deconv1 = self.__deconv__(512)
        self.deconv2 = self.__deconv__(256)
        self.deconv3 = self.__deconv__(128)
        self.deconv4 = self.__deconv__(128)

    def call(self, inputs):
        x1 = self.sharedDenseBlock01(self.conv1(inputs))
        x2 = MaxPooling2D(pool_size=(2, 2))(x1)
        x3 = self.sharedDenseBlock02(self.conv2(x2))
        x4 = MaxPooling2D(pool_size=(2, 2))(x3)
        x5 = self.sharedDenseBlock03(self.conv3(x4))
        x6 = MaxPooling2D(pool_size=(2, 2))(x5)
        x7 = self.sharedDenseBlock04(self.conv4(x6))
        x8 = MaxPooling2D(pool_size=(2, 2))(x7)
        x9 = self.denseBlock05(self.conv5(x8))

        z = self.denseBlock06(Concatenate(axis=-1)([self.deconv1(x9), x7]))
        z = self.denseBlock07(Concatenate(axis=-1)([self.deconv2(z), x5]))
        z = self.denseBlock08(Concatenate(axis=-1)([self.deconv3(z), x3]))
        z = self.denseBlock09(Concatenate(axis=-1)([self.deconv4(z), x1]))
        z = self.conv6(z)

        return z
