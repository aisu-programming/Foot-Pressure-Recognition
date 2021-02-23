

# foot_front = None
# heel_point = None

# 19 * 6
# 0 ~ 180 psi

''' Libraries '''
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Layer, ZeroPadding2D, Conv2D, MaxPooling2D, Reshape, Bidirectional, LSTM, Dense


''' Functions '''
class CustomizedConvolution(Layer):
    def __init__(self, units, **kwargs):
        super(CustomizedConvolution, self).__init__(**kwargs)
        self.units = units

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'units': self.units,
        })
        return config

    def build(self, input_shape):
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

def separativeConvolution():
    model = Sequential()
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
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    print('')
    model.summary()
    print('')
    return model


''' Execution '''
if __name__ == "__main__":
    CustomizedConvolution()