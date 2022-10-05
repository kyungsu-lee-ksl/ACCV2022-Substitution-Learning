from keras.layers import MaxPooling2D, Dense, Flatten

from model.BaseModel import BaseModel
from util.tf.layers import DenseBlock


class Architecture(BaseModel):

    def __init__(self):
        super().__init__()
        self.denseBlock05 = DenseBlock(64)

        self.conv1 = self.__conv__(64)
        self.conv2 = self.__conv__(128)
        self.conv3 = self.__conv__(256)
        self.conv4 = self.__conv__(512)
        self.conv5 = self.__conv__(64)

        self.dense01 = Dense(1000, activation='relu')
        self.dense02 = Dense(1000, activation='relu')
        self.dense03 = Dense(2, activation='softmax')

    def call(self, inputs):
        x = self.sharedDenseBlock01(self.conv1(inputs))
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = self.sharedDenseBlock02(self.conv2(x))
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = self.sharedDenseBlock03(self.conv3(x))
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = self.sharedDenseBlock04(self.conv4(x))
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = self.denseBlock05(self.conv5(x))
        z = Flatten()(x)
        z = self.dense01(z)
        z = self.dense02(z)
        z = self.dense03(z)

        return z
