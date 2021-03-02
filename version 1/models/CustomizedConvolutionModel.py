# foot_front = None
# heel_point = None

# 19 * 6
# 0 ~ 180 psi

# 1 4 1 4 1 1 4 1 (17)

''' Libraries '''
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Layer, Conv2D, MaxPooling2D, Dropout, Flatten, ZeroPadding2D, Dense


''' Functions '''
class OverallConvolutionBlock(Layer):
    def __init__(self, units, conv=2, wide=[6, 12, 24], dropout=0.0, **kwargs):
        super(OverallConvolutionBlock, self).__init__(**kwargs)
        self.units   = units
        self.conv    = conv
        self.wide    = wide
        self.dropout = dropout

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'units'  : self.units,
            'conv'   : self.conv,
            'wide'   : self.wide,
            'dropout': self.dropout,
        })
        return config

    def build(self, input_shape):
        self.layers = []
        for i in range(len(self.wide)):
            for _ in range(self.conv):
                self.layers.append(Conv2D(self.wide[i], 3, padding='same', activation='relu'))
            self.layers.append(MaxPooling2D(pool_size=(2, 2)))
            self.layers.append(Dropout(self.dropout))
        self.layers.append(Flatten())

    def call(self, inputs):
        x = inputs
        for layer in self.layers:
            x = layer(x)
        return x


class SegmentaryConvolutionBlock(Layer):
    def __init__(self, units, conv=2, wide=6, dropout=0.0, **kwargs):
        super(SegmentaryConvolutionBlock, self).__init__(**kwargs)
        self.units   = units
        self.conv    = conv
        self.wide    = wide
        self.dropout = dropout

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'units'  : self.units,
            'conv'   : self.conv,
            'wide'   : self.wide,
            'dropout': self.dropout,
        })
        return config

    def build(self, input_shape):
        
        number_block = []
        for _ in range(self.conv):
            number_block.append(Conv2D(self.wide, 3, padding='same', activation='relu'))
        number_block.append(MaxPooling2D(pool_size=(2, 2)))
        number_block.append(Dropout(self.dropout))

        segmentation_block = [ ZeroPadding2D(padding=((0, 1), (0, 0))) ]
        for layer in number_block:
            segmentation_block.append(layer)

        self.blocks = [ [
            [ segmentation_block, number_block, number_block, number_block ]
        ] * 19 ] * 6
        
    def call(self, inputs):
        x = inputs
        output = []
        for i in range(6):
            output_tmp = []
            for j in range(19):
                segmentation = x[:, i, j]
                number_1     = segmentation[:,  0: 6, :, :]
                number_2     = segmentation[:,  5:11, :, :]
                number_3     = segmentation[:, 11:17, :, :]
                for layer in self.blocks[i][j][0]: segmentation = layer(segmentation)
                for layer in self.blocks[i][j][1]: number_1     = layer(number_1)
                for layer in self.blocks[i][j][2]: number_2     = layer(number_2)
                for layer in self.blocks[i][j][3]: number_3     = layer(number_3)
                numbers = tf.concat([ number_1, number_2, number_3 ], axis=1)
                output_tmp.append(Flatten()(tf.concat([ segmentation, numbers ], axis=2)))
            output.append(tf.concat([ row for row in output_tmp ], axis=1))
        return tf.concat([ row for row in output ], axis=1)


class CustomizedConvolutionModel(Model):
    def __init__(self):
        super(CustomizedConvolutionModel, self).__init__()
        self.blocks = []
        for _ in range(7):
            self.blocks.append(OverallConvolutionBlock(1, wide=[9, 27, 81], dropout=0.2))
        for _ in range(5):
            self.blocks.append(SegmentaryConvolutionBlock(1, wide=6, dropout=0.2))

    def call(self, inputs):
        images = inputs
        for index in range(len(images)):
            images[index] = self.blocks[index](images[index])
        return images


''' Execution '''
if __name__ == "__main__":
    model = CustomizedConvolutionModel()
    model.compile(loss="mean_absolute_percentage_error", optimizer="adam")
    model.build()
    print('')
    model.summary()