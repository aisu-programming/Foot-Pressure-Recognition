''' Libraries '''
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Layer, ZeroPadding2D, BatchNormalization, Conv2D, MaxPooling2D, Flatten, Dropout, Dense


''' Functions '''
class LastLayer(Layer):
    def __init__(self, units=1):
        super(LastLayer, self).__init__()
        self.units = units
        
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'units': self.units,
        })
        return config

    def build(self, input_shape):
        pass

    def call(self, inputs):
        return tf.multiply(inputs, tf.constant([120, 400, 120, 400], dtype='float32'))

def ClassicalConvolutionModel(image_amount):
    
    channel = image_amount * 3
    initializer = tf.keras.initializers.TruncatedNormal(mean=0., stddev=0.5)
    
    model = Sequential()
    model.add(Input(shape=(400, 120, channel)))
    # model.add(BatchNormalization(axis=1))
    model.add(Conv2D(9, 3, padding='same', activation='relu'))
    model.add(Conv2D(9, 3, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(27, 3, padding='same', activation='relu'))
    model.add(Conv2D(27, 3, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(81, 3, padding='same', activation='relu'))
    model.add(Conv2D(81, 3, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(BatchNormalization(axis=-1))
    model.add(Dense(800, kernel_initializer=initializer, bias_initializer=initializer, activation='relu'))
    model.add(Dense(400, kernel_initializer=initializer, bias_initializer=initializer, activation='relu'))
    model.add(Dense(200, kernel_initializer=initializer, bias_initializer=initializer, activation='relu'))
    model.add(Dropout(0.2))
    model.add(BatchNormalization(axis=-1))
    model.add(Dense(100, kernel_initializer=initializer, bias_initializer=initializer, activation='relu'))
    model.add(Dense(50, kernel_initializer=initializer, bias_initializer=initializer, activation='relu'))
    model.add(Dense(4, kernel_initializer=initializer, bias_initializer=initializer, activation='relu'))

    return model