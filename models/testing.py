''' Libraries '''
import os
from tensorflow.keras import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import ZeroPadding2D, Conv2D, MaxPooling2D, Flatten, Dropout, Dense

from ClassicalConvolutionModel import ClassicalConvolutionModel


''' Functions '''
def test_ClassicalConvoluntionModel():
    model = ClassicalConvolutionModel(image_amount=3)
    print('')
    model.summary()


def test_OverallConvoluntionBlock():
    model = Sequential()
    
    # model.add(Input(shape=(120, 400, 3)))
    # model.add(Conv2D(6, (7, 29), activation='relu'))
    # model.add(Conv2D(6, (7, 29), activation='relu'))
    # model.add(Conv2D(6, (7, 29), activation='relu'))

    model.add(Input(shape=(102, 316, 3)))
    model.add(Conv2D(6, (3, 2), activation='relu'))
    model.add(Conv2D(6, (3, 2), activation='relu'))
    model.add(Conv2D(6, (3, 2), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(12, (3, 2), activation='relu'))
    model.add(Conv2D(12, (3, 2), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(24, 3, activation='relu'))
    model.add(Conv2D(24, 3, activation='relu'))
    model.add(Conv2D(24, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 3)))
    model.add(Flatten())
    print('')
    model.summary()


def test_SegmentaryConvolutionBlock():
    model = Sequential()
    model.add(Input(shape=(17, 10, 3)))
    model.add(Conv2D(6, 3, activation='relu'))
    model.add(Conv2D(6, 3, activation='relu'))
    model.add(Conv2D(12, 3, activation='relu'))
    model.add(Conv2D(12, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    print('')
    model.summary()


''' Execution '''
if __name__ == '__main__':
    os.system('cls')
    # test_ClassicalConvoluntionModel()
    test_OverallConvoluntionBlock()
    test_SegmentaryConvolutionBlock()