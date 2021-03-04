# foot_front = None
# heel_point = None

# 19 * 6
# 0 ~ 180 psi

# 1 4 1 4 1 1 4 1 (17)


''' Parameters '''
PRINT_MAIN_STEP_TIME = False
PRINT_CONVOLUTION_STEP_TIME = False


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
from tensorflow.keras.layers import Layer, BatchNormalization, Conv2D, MaxPooling2D, Dropout, Flatten, ZeroPadding2D, Dense

physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) != 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


''' Functions '''
class RawSizeConvolutionBlock(Layer):
    def __init__(self, conv, conv_units, denses, dropout, **kwargs):
        super(RawSizeConvolutionBlock, self).__init__(**kwargs)
        self.conv    = conv
        self.conv_units   = conv_units
        self.denses  = denses
        self.dropout = dropout

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'conv'      : self.conv,
            'conv_units': self.conv_units,
            'denses'    : self.denses,
            'dropout'   : self.dropout,
        })
        return config

    def build(self, input_shape):
        self.layers = []
        for i in range(len(self.conv_units)):
            for _ in range(self.conv):
                self.layers.append(Conv2D(self.conv_units[i], 3, padding='same', activation='relu'))
            self.layers.append(MaxPooling2D(pool_size=(2, 2)))
            self.layers.append(Dropout(self.dropout))
        self.layers.append(Flatten())
        self.layers.append(BatchNormalization(axis=-1))
        for dense in self.denses:
            self.layers.append(Dense(dense, activation='relu'))
        self.layers.append(Dropout(self.dropout))
        self.layers.append(BatchNormalization(axis=-1))

    def call(self, inputs):
        # Input:              (BATCH_SIZE, 400, 120,  3)
        # Conv2D * self.conv: (BATCH_SIZE, 400, 120,  self.conv_units[0])
        # MaxPooling2D:       (BATCH_SIZE, 200,  60,  self.conv_units[0])
        # Conv2D * self.conv: (BATCH_SIZE, 200,  60,  self.conv_units[1])
        # MaxPooling2D:       (BATCH_SIZE, 100,  30,  self.conv_units[1])
        # Conv2D * self.conv: (BATCH_SIZE, 100,  30,  self.conv_units[2])
        # MaxPooling2D:       (BATCH_SIZE,  50,  15,  self.conv_units[2])
        # Flatten:            (BATCH_SIZE,  50 * 15 * self.conv_units[2])  # (BATCH_SIZE, 50*15*24) = (BATCH_SIZE, 18000)
        # Dense:              (BATCH_SIZE, self.denses[0])                 # (BATCH_SIZE, 18000) ---> (BATCH_SIZE, 30)
        # Dense:              (BATCH_SIZE, self.denses[:])
        # Output:             (BATCH_SIZE, self.denses[-1])                # (BATCH_SIZE, 1600)
        x = inputs
        for layer in self.layers:
            x = layer(x)
        return x


class CroppedConvolutionBlock(Layer):
    def __init__(self, conv, conv_units, denses, dropout, **kwargs):
        super(CroppedConvolutionBlock, self).__init__(**kwargs)
        self.conv    = conv
        self.conv_units   = conv_units
        self.denses  = denses
        self.dropout = dropout

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'conv'      : self.conv,
            'conv_units': self.conv_units,
            'denses'    : self.denses,
            'dropout'   : self.dropout,
        })
        return config

    def build(self, input_shape):
        self.layers = []
        for i in range(len(self.conv_units)):
            if   i == 0: self.layers.append(ZeroPadding2D(padding=(0, 1)))  # (height, width)
            elif i == 1: self.layers.append(ZeroPadding2D(padding=(1, 0)))  # (height, width)
            for _ in range(self.conv):
                self.layers.append(Conv2D(self.conv_units[i], 3, padding='same', activation='relu'))
            self.layers.append(MaxPooling2D(pool_size=(2, 2)))
            self.layers.append(Dropout(self.dropout))
        self.layers.append(Flatten())
        self.layers.append(BatchNormalization(axis=-1))
        for dense in self.denses:
            self.layers.append(Dense(dense, activation='relu'))
        self.layers.append(Dropout(self.dropout))
        self.layers.append(BatchNormalization(axis=-1))

    def call(self, inputs):
        # Input:              (BATCH_SIZE, 316, 102,  3)
        # ZeroPadding2D:      (BATCH_SIZE, 316, 104,  3)
        # Conv2D * self.conv: (BATCH_SIZE, 316, 104,  self.conv_units[0])
        # MaxPooling2D:       (BATCH_SIZE, 158,  52,  self.conv_units[0])
        # ZeroPadding2D:      (BATCH_SIZE, 160,  52,  self.conv_units[0])
        # Conv2D * self.conv: (BATCH_SIZE, 160,  52,  self.conv_units[1])
        # MaxPooling2D:       (BATCH_SIZE,  80,  26,  self.conv_units[1])
        # Conv2D * self.conv: (BATCH_SIZE,  80,  26,  self.conv_units[2])
        # MaxPooling2D:       (BATCH_SIZE,  40,  13,  self.conv_units[2])
        # Flatten:            (BATCH_SIZE,  40 * 13 * self.conv_units[2])  # (BATCH_SIZE, 40*13*24) = (BATCH_SIZE, 12480)
        # Dense:              (BATCH_SIZE, self.denses[0])                 # (BATCH_SIZE, 12480) ---> (BATCH_SIZE, 30)
        # Dense:              (BATCH_SIZE, self.denses[:])
        # Output:             (BATCH_SIZE, self.denses[-1])                # (BATCH_SIZE, 1600)
        x = inputs
        for layer in self.layers:
            x = layer(x)
        return x


class SegmentaryConvolutionBlock(Layer):
    def __init__(self, conv, conv_unit, conv_denses, integrating_denses, dropout, **kwargs):
        super(SegmentaryConvolutionBlock, self).__init__(**kwargs)
        self.conv               = conv
        self.conv_unit          = conv_unit
        self.conv_denses        = conv_denses
        self.integrating_denses = integrating_denses
        self.dropout            = dropout

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'conv'              : self.conv,
            'conv_unit'         : self.conv_unit,
            'conv_denses'       : self.conv_denses,
            'integrating_denses': self.integrating_denses,
            'dropout'           : self.dropout,
        })
        return config

    def build(self, input_shape):
        
        # Build number blocks
        number_block = []
        for _ in range(self.conv):
            number_block.append(Conv2D(self.conv_unit, 3, padding='same', activation='relu'))
        number_block.append(MaxPooling2D(pool_size=(2, 2)))
        number_block.append(Dropout(self.dropout))
        number_block.append(Flatten())
        number_block.append(BatchNormalization(axis=-1))
        for dense in self.conv_denses:
            number_block.append(Dense(dense, activation='relu'))
        number_block.append(Dropout(self.dropout))
        number_block.append(BatchNormalization(axis=-1))
        self.number_blocks = [ number_block, number_block, number_block ]

        # Build segmentation block
        self.segmentation_block = [ ZeroPadding2D(padding=((0, 0), (0, 1))) ]
        for _ in range(self.conv):
            self.segmentation_block.append(Conv2D(self.conv_unit, 3, padding='same', activation='relu'))
        self.segmentation_block.append(MaxPooling2D(pool_size=(2, 2)))
        self.segmentation_block.append(Dropout(self.dropout))
        self.segmentation_block.append(Flatten())
        self.segmentation_block.append(BatchNormalization(axis=-1))
        for dense in self.conv_denses:
            self.segmentation_block.append(Dense(dense, activation='relu'))
        self.segmentation_block.append(Dropout(self.dropout))
        self.segmentation_block.append(BatchNormalization(axis=-1))
        
        # Build integrating block
        self.integrating_block = [ Dense(dense, activation='relu') for dense in self.integrating_denses ]
        self.integrating_block.append(Dropout(self.dropout))
        self.segmentation_block.append(BatchNormalization(axis=-1))
        
    def call(self, inputs):
        x = inputs
        output_batchs = []
        for bi in range(x.shape[0]):  # bi = batch_index
            segmentation = tf.reshape(x[bi], [114, x.shape[3], x.shape[4], x.shape[5]])
            number_1     = segmentation[:, :,  0: 6, :]
            number_2     = segmentation[:, :,  5:11, :]
            number_3     = segmentation[:, :, 11:17, :]

            # Segmentation input:  (114, 10, 17,  3)
            # ZeroPadding2D:       (114, 10, 18,  3)
            # Conv2D * self.conv:  (114, 10, 18,  self.conv_unit)
            # MaxPooling2D:        (114,  5,  9,  self.conv_unit)
            # Flatten:             (114,  5 * 9 * self.conv_unit)  # (114, 5*9*9) = (114, 405)
            # Dense:               (114, self.conv_denses[0])      # (114, 405) --> (114, 100) 
            # Dense:               (114, self.conv_denses[:])
            # Segmentation output: (114, self.conv_denses[-1])     # (114, 30)
            for layer in self.segmentation_block: segmentation = layer(segmentation)

            # Number input:       (114, 10,  6,  3)
            # Conv2D * self.conv: (114, 10,  6,  self.conv_unit)
            # MaxPooling2D:       (114,  5,  3,  self.conv_unit)
            # Flatten:            (114,  5 * 3 * self.conv_unit)  # (114, 5*3*9) = (114, 135)
            # Dense:              (114, self.conv_denses[0])      # (114, 135) --> (114, 100)
            # Dense:              (114, self.conv_denses[:])
            # Number output:      (114, self.conv_denses[-1])     # (114, 40)
            for layer in self.number_blocks[0]: number_1 = layer(number_1)
            for layer in self.number_blocks[1]: number_2 = layer(number_2)
            for layer in self.number_blocks[2]: number_3 = layer(number_3)
            
            concatenation = tf.concat([ segmentation, number_1, number_2, number_3 ], axis=1)  # (114, 160)
            # concatenation = segmentation  # (114, 30)
            for layer in self.integrating_block: concatenation = layer(concatenation)          # (114,  20)
            concatenation = tf.reshape(concatenation, [ 114 * self.integrating_denses[-1] ])   # (114 * 20) = (2280)
            output_batchs.append(concatenation)

        return tf.convert_to_tensor(output_batchs)  # (BATCH_SIZE, 2280)


class CustomizedConvolutionModel(Model):
    def __init__(self):
        super(CustomizedConvolutionModel, self).__init__()
        self.convolution_blocks = []
        for _ in range(2):
            self.convolution_blocks.append(RawSizeConvolutionBlock(
                conv=3, conv_units=[6, 12, 24], denses=[40, 300, 900, 1600, 2200], dropout=0.2))
        for _ in range(3):
            self.convolution_blocks.append(CroppedConvolutionBlock(
                conv=3, conv_units=[6, 12, 24], denses=[40, 300, 900, 1600, 2200], dropout=0.2))
        for _ in range(3):
            self.convolution_blocks.append(SegmentaryConvolutionBlock(
                conv=3, conv_unit=9, conv_denses=[110, 80, 50], integrating_denses=[160, 110, 60, 20], dropout=0.2))

        self.normalizing_blocks = []
        for _ in range(8):
            self.normalizing_blocks.append([
                Dense(1600, activation='relu'),
                Dense(800, activation='relu'),
                Dense(200, activation='relu'),
                Dense(100, activation='relu'),
                Dropout(0.2),
                BatchNormalization(axis=-1),
            ])
        self.integrating_block = [ 
            Dense(800, activation='relu'),
            Dense(200, activation='relu'),
            Dense(50, activation='relu'),
            Dense(20, activation='relu'),
            BatchNormalization(axis=-1),
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
        # print(cropped_im)  # (BATCH_SIZE, 316, 102, 3)
        # Image.fromarray(images[:, 2, 42:358, 9:111, :].numpy()[0]).save('test.png')
        sharpen_im     = tf.dtypes.cast(self.__image_filter(cropped_im, ImageFilter.SHARPEN), tf.float32)
        contour_im     = tf.dtypes.cast(self.__image_filter(cropped_im, ImageFilter.CONTOUR), tf.float32)

        cropped_im_seg = tf.dtypes.cast(self.__segment_image(cropped_im), tf.float32)
        sharpen_im_seg = tf.dtypes.cast(self.__segment_image(sharpen_im), tf.float32)
        contour_im_seg = tf.dtypes.cast(self.__segment_image(contour_im), tf.float32)
        # # print(cropped_im_seg)  # (BATCH_SIZE, 6, 19, 10, 17, 3)

        images = [
            original_im, color_im,                           # RawSizeConvolutionBlock
            cropped_im, sharpen_im, contour_im,              # CroppedConvolutionBlock
            cropped_im_seg, sharpen_im_seg, contour_im_seg,  # SegmentaryConvolutionBlock
        ]

        if PRINT_MAIN_STEP_TIME: start = time.time()
        for index in range(len(images)):
            if PRINT_CONVOLUTION_STEP_TIME: start = time.time()
            images[index] = self.convolution_blocks[index](images[index])
            if PRINT_CONVOLUTION_STEP_TIME: print(f'Convolution blocks {index}: ', time.time() - start)
        if PRINT_MAIN_STEP_TIME: print('Convolution blocks: ', time.time() - start)

        if PRINT_MAIN_STEP_TIME: start = time.time()
        for index in range(len(images)):
            for layer in self.normalizing_blocks[index]:
                images[index] = layer(images[index])
        if PRINT_MAIN_STEP_TIME: print('Normalizing blocks: ', time.time() - start)

        if PRINT_MAIN_STEP_TIME: start = time.time()
        images = tf.concat(images, axis=1)
        for layer in self.integrating_block:
            images = layer(images)
        if PRINT_MAIN_STEP_TIME: print('Integrating block: ', time.time() - start)

        return images