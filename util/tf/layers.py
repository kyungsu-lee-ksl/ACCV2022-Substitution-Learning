from tensorflow.keras import layers
from tensorflow.keras.layers import Conv2D, Conv3D, Concatenate


class DenseBlock(layers.Layer):
    def __init__(self, k):
        super(DenseBlock, self).__init__()
        self.layers = []

        dilation_rate = 6 * (k // 64)

        self.ast01 = self.__atrous_conv__(k, dilation_rate=dilation_rate)
        self.ast02 = self.__atrous_conv__(k // 8, dilation_rate=dilation_rate)

        self.conv01 = self.__conv__(k)
        self.ast03 = self.__atrous_conv__(k, dilation_rate=dilation_rate)
        self.ast04 = self.__atrous_conv__(k // 8, dilation_rate=dilation_rate)

        self.conv02 = self.__conv__(k)
        self.ast05 = self.__atrous_conv__(k, dilation_rate=dilation_rate)
        self.ast06 = self.__atrous_conv__(k // 8, dilation_rate=dilation_rate)

        self.conv03 = self.__conv__(k)
        self.ast07 = self.__atrous_conv__(k, dilation_rate=dilation_rate)
        self.ast08 = self.__atrous_conv__(k // 8, dilation_rate=dilation_rate)

    def __conv__(self, output_channel, activation='relu'):
        return Conv2D(output_channel, (1, 1), activation=activation, padding="SAME")

    def __atrous_conv__(self, output_channel, dilation_rate=6, activation='relu'):
        return Conv2D(output_channel, (3, 3), dilation_rate=dilation_rate, activation=activation, padding="SAME")

    def call(self, inputs):

        k = int(inputs.shape[-1])
        assert k % 8 == 0, f"k={k}"

        # l=1
        layer00 = inputs
        layer01 = self.ast01(layer00)
        layer02 = self.ast02(layer01)

        # l=2
        layer10 = Concatenate(axis=-1)([layer00, layer02])
        layer11 = self.conv01(layer10)
        layer12 = self.ast03(layer11)
        layer13 = self.ast04(layer12)

        # l=3
        layer20 = Concatenate(axis=-1)([layer00, layer02, layer13])
        layer21 = self.conv02(layer20)
        layer22 = self.ast05(layer21)
        layer23 = self.ast06(layer22)

        # l=4
        layer30 = Concatenate(axis=-1)([layer00, layer02, layer13, layer23])
        layer31 = self.conv03(layer30)
        layer32 = self.ast07(layer31)
        layer33 = self.ast08(layer32)

        self.layers.extend([
            layer00, layer01, layer02,
            layer10, layer11, layer12, layer13,
            layer20, layer21, layer22, layer23,
            layer30, layer31, layer32, layer33,
        ])

        return layer33
