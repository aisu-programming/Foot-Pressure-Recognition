''' Libraries '''
import os
from tensorflow.keras import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import ZeroPadding2D, Conv2D, MaxPooling2D, Flatten, Dropout, Dense


''' Functions '''
def ClassicalConvoluntionModel(image_amount):
    channel = image_amount * 3
    
    model = Sequential()
    model.add(Input(shape=(316, 102, channel)))
    model.add(Conv2D(channel*2, 3, activation='relu'))
    model.add(Conv2D(channel*2, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(channel*4, 3, activation='relu'))
    model.add(Conv2D(channel*4, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(channel*8, 3, activation='relu'))
    model.add(Conv2D(channel*8, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(400, activation='relu'))
    model.add(Dense(400, activation='sigmoid'))
    model.add(Dense(200, activation='relu'))
    model.add(Dense(200, activation='sigmoid'))
    model.add(Dropout(0.2))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(100, activation='sigmoid'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(50, activation='sigmoid'))
    model.add(Dropout(0.2))
    model.add(Dense(30, activation='relu'))
    model.add(Dense(30, activation='sigmoid'))
    model.add(Dense(15, activation='relu'))
    model.add(Dense(4))
    model.compile(loss="mean_absolute_percentage_error", optimizer="adam")

    print('')
    model.summary()
    print('')
    return model


''' Execution '''
if __name__ == '__main__':
    os.system('cls')
    ClassicalConvoluntionModel(image_amount=7)