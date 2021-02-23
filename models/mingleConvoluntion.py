''' Libraries '''
import os
from tensorflow.keras import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import ZeroPadding2D, Conv2D, MaxPooling2D, Flatten, Dense


''' Functions '''
def mingleConvoluntion(image_amount):
    model = Sequential()
    model.add(Input(shape=(316, 102, image_amount*3)))
    model.add(Conv2D(48, 3, activation='relu'))
    model.add(Conv2D(48, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(36, 3, activation='relu'))
    model.add(Conv2D(36, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(24, 3, activation='relu'))
    model.add(Conv2D(24, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(4))
    model.compile(loss="mean_absolute_percentage_error", optimizer="adam")
    print('')
    model.summary()
    print('')
    return model


''' Execution '''
if __name__ == '__main__':
    os.system('cls')
    mingleConvoluntion(image_amount=7)