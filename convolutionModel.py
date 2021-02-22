train_ignoring   = [ 67,  83, 170, 171, 235, 245, 313, 373, 404, 410, 478, 589, 600, 609, 625, 637, 649, 656, 662, 685, 749, 751, 797, 801, 826, 872, 891, 933, 953, 980]
test_malfunction = [ 91,  95, 111, 153]

# foot_front = None
# heel_point = None

# 19 * 6
# 0 ~ 180 psi

''' Libraries '''
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import ZeroPadding2D, Conv2D, MaxPooling2D, Reshape, Bidirectional, LSTM, Dense


''' Functions '''
def ConvolutionModel():
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
    ConvolutionModel()