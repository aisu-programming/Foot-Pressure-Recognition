

# foot_front = None
# heel_point = None

# 19 * 6
# 0 ~ 180 psi

''' Libraries '''
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Layer, ZeroPadding2D, Conv2D, MaxPooling2D, Reshape, Bidirectional, LSTM, Dense


''' Functions '''
class CustomizedLayer(Layer):
    def __init__(self, units=1, images=[], original=False, **kwargs):
        super(CustomizedLayer, self).__init__(**kwargs)
        self.units         = units
        self.images        = images
        self.original      = original

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'units'        : self.units,
            'images'       : self.images,
            'original'     : self.original,
        })
        return config

    def build(self):
        layers_for_others   = [
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
        ]

        layers_for_original = [
            Conv2D(6, (3, 5), activation='relu'),
            Conv2D(6, (3, 5), activation='relu'),
            Conv2D(6, (3, 5), activation='relu'),
            layer for layer in layers_for_others
        ]

        if original:
            self.convolutions = [
                Conv2D(len(self.images)*2, 3, activation='relu') for _ in range(len(self.images))
            ]
        else:
            self.convolutions = [
                Conv2D(len(self.images)*2, 3, activation='relu') for _ in range(len(self.images))
            ]

        self.layers = [
            Conv1D(filters=self.units, kernel_size=3, activation='relu'), 
            ZeroPadding1D(padding=(1, 1)), 
            Dropout(0.2), 
            Conv1D(filters=self.units, kernel_size=3), 
            ZeroPadding1D(padding=(1, 1)), 
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

    model.add(ZeroPadding2D(padding=1, input_shape=(316, 102, 3)))
    model.add(Conv2D(16, 3, activation='relu'))
    model.add(ZeroPadding2D(padding=1))
    model.add(Conv2D(16, 3, activation='relu'))
    model.add(ZeroPadding2D(padding=1))
    model.add(Conv2D(16, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(ZeroPadding2D(padding=1))
    model.add(Conv2D(32, 3, activation='relu'))
    model.add(ZeroPadding2D(padding=1))
    model.add(Conv2D(32, 3, activation='relu'))
    model.add(ZeroPadding2D(padding=1))
    model.add(Conv2D(32, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(ZeroPadding2D(padding=1))
    model.add(Conv2D(64, 3, activation='relu'))
    model.add(ZeroPadding2D(padding=1))
    model.add(Conv2D(64, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    # model.add(Reshape((-1, 64)))
    # model.add(Bidirectional(LSTM(50, input_length=10, input_dim=64)))
    model.add(Dense(5, activation='softmax'))
    model.compile(loss="categorical_crossentropy", optimizer="adam")

    print('')
    model.summary()
    print('')
    return model


''' Execution '''
if __name__ == "__main__":
    CustomizedConvolution()