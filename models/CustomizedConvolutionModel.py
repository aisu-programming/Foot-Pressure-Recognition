

# foot_front = None
# heel_point = None

# 19 * 6
# 0 ~ 180 psi

''' Libraries '''
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Layer, ZeroPadding2D, Conv2D, MaxPooling2D, Flatten, Dense


''' Functions '''
class CroppedConvolution(Layer):
    def __init__(self, units, shape, **kwargs):
        super(CustomizedLayer, self).__init__(**kwargs)
        self.units = units
        self.shape = shape

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'units': self.units,
            'shape': self.shape,
        })
        return config

    def build(self):
        self.layers = [
            Conv2D(12, 3, activation='relu'),
            Conv2D(12, 3, activation='relu'),
            Conv2D(12, 3, activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(24, 3, activation='relu'),
            Conv2D(24, 3, activation='relu'),
            Conv2D(24, 3, activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(48, 3, activation='relu'),
            Conv2D(48, 3, activation='relu'),
            Conv2D(48, 3, activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dropout(0.2),
            Dense(200, activation='relu'),
            Dense(200, activation='sigmoid'),
            Dense(100, activation='relu'),
            Dense(100, activation='sigmoid'),
            Dropout(0.2),
            Dense(50, activation='relu'),
            Dense(50, activation='sigmoid'),
            Dense(30, activation='relu'),
            Dense(30, activation='sigmoid'),
        ]


class CustomizedLayer(Layer):
    def __init__(self, units=1, images=[], original=False, **kwargs):
        super(CustomizedLayer, self).__init__(**kwargs)
        self.units    = units
        self.images   = images
        self.original = original

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'units'   : self.units,
            'images'  : self.images,
            'original': self.original,
        })
        return config

    def build(self):

        layers_for_cropped = [
            
        ]

        # 120*400 -> 102*316 (-18/-84)
        layers_for_raw_size = [
            Conv2D(6, (7, 57), activation='relu'),  # -6/-28
            Conv2D(6, (7, 57), activation='relu'),  # -6/-28
            Conv2D(6, (7, 57), activation='relu'),  # -6/-28
            Dropout(0.2),
            layer for layer in layers_for_cropped
        ]

        if original:
            self.layers = [
                layers_for_raw_size, layers_for_raw_size, 
                layers_for_cropped for _ in range(len(self.images)-2)
            ]
        else:
            self.layers = [
                layers_for_raw_size, 
                layers_for_cropped for _ in range(len(self.images-1))
            ]

    def call(self, inputs):
        x = inputs
        for layer in self.layers:
            x = layer(x)
        return x


def separativeConvolution(image_amount):
    channel = image_amount * 3

    model = Sequential()
    model.add(Input(shape=(316, 102, channel)))

    model.add(CustomizedLayer())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(100, activation='sigmoid'))
    model.add(Dense(30, activation='relu'))
    model.add(Dense(30, activation='sigmoid'))
    model.add(Dense(4))
    model.compile(loss="mean_absolute_percentage_error", optimizer="adam")

    print('')
    model.summary()
    print('')
    return model


''' Execution '''
if __name__ == "__main__":
    CustomizedConvolution()