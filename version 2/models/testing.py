''' Libraries '''
import os
from tensorflow.keras import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import ZeroPadding2D, Conv2D, MaxPooling2D, Flatten, Dropout, Dense

from ClassicalConvolutionModel import ClassicalConvolutionModel
from CustomizedConvolutionModel import OverallConvolutionBlock, SegmentaryConvolutionBlock, CustomizedConvolutionModel


''' Functions '''
def test_ClassicalConvoluntionModel():
    model = ClassicalConvolutionModel(image_amount=3)
    print('')
    model.summary()


def test_OverallConvolutionBlock():
    model = Sequential()
    model.add(Input(shape=(120, 400, 3)))
    model.add(OverallConvolutionBlock(1, wide=[9, 27, 81], dropout=0.1))
    model.compile(loss="mean_absolute_percentage_error", optimizer="adam")
    print('')
    model.summary()


def test_SegmentaryConvolutionBlock():
    model = Sequential()
    model.add(Input(shape=(6, 19, 17, 10, 3)))
    model.add(SegmentaryConvolutionBlock(1, dropout=0.1))
    model.compile(loss="mean_absolute_percentage_error", optimizer="adam")
    print('')
    model.summary()


def test_CustomizedConvolutionModel():
    model = CustomizedConvolutionModel()
    model.compile(loss="mean_absolute_percentage_error", optimizer="adam")
    model.build()
    print('')
    model.summary()


''' Execution '''
if __name__ == '__main__':
    os.system('cls')
    test_CustomizedConvolutionModel()