# foot_front = None
# heel_point = None

# 19 * 6
# 0 ~ 180 psi

# 1 4 1 4 1 1 4 1 (17)


''' Parameters '''
DEBUG = False


''' Libraries '''
# Programming
import time

# Data processing
import math
import numpy as np
from PIL import Image, ImageFilter

# Deep learning
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Layer, Conv2D, MaxPooling2D, Dropout, Flatten, ZeroPadding2D, Dense

physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)


''' Functions '''
class OverallConvolutionBlock(Layer):
    def __init__(self, conv, wide, dropout, **kwargs):
        super(OverallConvolutionBlock, self).__init__(**kwargs)
        self.conv    = conv
        self.wide    = wide
        self.dropout = dropout

    def get_config(self):
        config = super().get_config().copy()
        config.update({
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
    def __init__(self, conv, wide, dropout, **kwargs):
        super(SegmentaryConvolutionBlock, self).__init__(**kwargs)
        self.conv    = conv
        self.wide    = wide
        self.dropout = dropout

    def get_config(self):
        config = super().get_config().copy()
        config.update({
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
        # number_block.append(Dense())

        segmentation_block = [ ZeroPadding2D(padding=((0, 0), (0, 1))) ]
        for layer in number_block:
            segmentation_block.append(layer)

        self.blocks = [ segmentation_block, number_block, number_block, number_block ]
        
    def call(self, inputs):
        x = inputs

        output_batchs = []
        for bi in range(x.shape[0]):  # bi = batch_index
            segmentation = tf.reshape(x[bi], [114, x.shape[3], x.shape[4], x.shape[5]])
            number_1     = segmentation[:, :,  0: 6, :]
            number_2     = segmentation[:, :,  5:11, :]
            number_3     = segmentation[:, :, 11:17, :]

            # (BATCH_SIZE, 10, 17, 3) --> (BATCH_SIZE, 5, 9, 3)
            for layer in self.blocks[0]: segmentation = layer(segmentation)
            # (BATCH_SIZE, 10,  6, 3) --> (BATCH_SIZE, 5, 3, 3)
            for layer in self.blocks[1]: number_1     = layer(number_1)
            for layer in self.blocks[2]: number_2     = layer(number_2)
            for layer in self.blocks[3]: number_3     = layer(number_3)


            
            numbers = tf.concat([ number_1, number_2, number_3 ], axis=2)
            segmentation = tf.concat([ segmentation, numbers ], axis=1)
            segmentation = tf.reshape(segmentation, 10260*self.wide)
            output_batchs.append(segmentation)

        return tf.convert_to_tensor(output_batchs)


class CustomizedConvolutionModel(Model):
    def __init__(self):
        super(CustomizedConvolutionModel, self).__init__()
        self.blocks = []
        for _ in range(5):
            self.blocks.append(OverallConvolutionBlock(conv=3, wide=[9, 27, 81], dropout=0.2))
        for _ in range(3):
            self.blocks.append(SegmentaryConvolutionBlock(conv=3, wide=9, dropout=0.2))
        self.integrating_layers = []
        for _ in range(8):
            self.integrating_layers.append([
                Dense(300, activation='relu'),
                Dense(300, activation='relu'),
                Dense(300, activation='relu'),
                Dense(300, activation='relu'),
                Dropout(0.2),
            ])
        self.integrating_ending_layer = [ 
            Dense(200, activation='relu'),
            Dense(100, activation='relu'),
            Dense(50, activation='relu'),
            Dense(20, activation='relu'),
            Dense(4, activation='relu'),
        ]

    def __image_filter(self, image_tensors, filter):
        image_numpys = image_tensors.numpy()
        image_output = []
        for image_numpy in image_numpys:
            filtered_image = tf.keras.preprocessing.image.array_to_img(image_numpy).filter(filter)
            reshaped_filtered_image_numpy = np.array(filtered_image).reshape(316, 102, 3)
            image_output.append(reshaped_filtered_image_numpy)
        return tf.convert_to_tensor(np.array(image_output))

    def __segment_image(self, image_tensors):
        image_numpys = image_tensors.numpy()
        image_output = []
        for image_numpy in image_numpys:
            segmentations = []
            for w in range(6):
                segmentation_row = []
                for h in range(19):
                    left   = int(17 * w)
                    top    = int((316 + 7) / 19 * h)
                    segmentation_row.append(image_numpy[top:top+10, left:left+17, :])
                segmentations.append(segmentation_row)
            image_output.append(segmentations)
        return tf.convert_to_tensor(np.array(image_output))

    def call(self, inputs):
        images = inputs

        original_im    = tf.dtypes.cast(images[:, 0], tf.float32)
        color_im       = tf.dtypes.cast(images[:, 1], tf.float32)
        # print(color_im)  # (BATCH_SIZE, 400, 120, 3)

        cropped_im     = tf.dtypes.cast(images[:, 2, 42:358, 9:111, :], tf.float32)
        # print(cropped_im.numpy()[0].shape)
        # Image.fromarray(images[:, 2, :316, :102, :].numpy()[0]).save('test.png')
        sharpen_im     = tf.dtypes.cast(self.__image_filter(cropped_im, ImageFilter.SHARPEN), tf.float32)
        contour_im     = tf.dtypes.cast(self.__image_filter(cropped_im, ImageFilter.CONTOUR), tf.float32)
        # print(cropped_im)  # (BATCH_SIZE, 316, 102, 3)

        cropped_im_seg = tf.dtypes.cast(self.__segment_image(cropped_im), tf.float32)
        sharpen_im_seg = tf.dtypes.cast(self.__segment_image(sharpen_im), tf.float32)
        contour_im_seg = tf.dtypes.cast(self.__segment_image(contour_im), tf.float32)
        # print(cropped_im_seg)  # (BATCH_SIZE, 6, 19, 10, 17, 3)

        images = [
            original_im, color_im,
            cropped_im, sharpen_im, contour_im,
            cropped_im_seg, sharpen_im_seg, contour_im_seg,
        ]

        if DEBUG: start = time.time()
        for index in range(len(images)):
            images[index] = self.blocks[index](images[index])
            for layer in self.integrating_layers[index]:
                images[index] = layer(images[index])
        if DEBUG: print('train 1: ', time.time() - start)

        if DEBUG: start = time.time()
        images = tf.concat(images, axis=1)
        for layer in self.integrating_ending_layer:
            images = layer(images)
        if DEBUG: print('train 2: ', time.time() - start)

        return images


''' Execution '''
if __name__ == "__main__":
    model = CustomizedConvolutionModel()
    model.compile(loss="mean_absolute_percentage_error", optimizer="adam")
    model.build()
    print('')
    model.summary()