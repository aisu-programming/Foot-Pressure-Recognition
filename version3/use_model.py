''' Libraries '''
import os
from tensorflow.keras.models import load_model, load_weights
from train import my_euclidean_distance_loss
from models.CustomizedConvolutionModel import RawSizeConvolutionBlock, CroppedConvolutionBlock, SegmentaryConvolutionBlock, CustomizedConvolutionModel


''' Parameters '''
# MODEL = 'ClassicalConvolutionModel'
MODEL = 'CustomizedConvolutionModel'


''' Functions '''
def print_model(model):
    if MODEL == 'CustomizedConvolutionModel':
        print("Can't print CustomizedConvolutionModel.")
        return
    model.summary()
    return


def read_test_data():
    print('')
    X = []
    for i in tqdm(range(1, 1001), desc=f"Reading testing images", ascii=True):
        original_im = np.array(Image.open(f"processed_test_data/image_{i:04d}_original.png"))
        color_im    = np.array(Image.open(f"processed_test_data/image_{i:04d}_color.png"))
        cropped_im  = np.array(Image.open(f"processed_test_data/image_{i:04d}_cropped.png"))
        if MODEL == 'ClassicalConvolutionModel':
            X.append(np.concatenate([
                original_im, color_im, cropped_im,
            ], axis=2))
        elif MODEL == 'CustomizedConvolutionModel':
            X.append([ original_im, color_im, cropped_im ])
    X = np.array(X)
    print('\n')
    return X


def predict_by_model(model):
    Y_pred = model.predict(read_test_data())


''' Execution '''
if __name__ == '__main__':

    os.system('cls')

    model_path = r"..\output\2021-03-03\21.10.45-ver3-customized-20.01\best_model_min_val_loss.h5"
    custom_objects = {
        'my_euclidean_distance_loss': my_euclidean_distance_loss,
        'RawSizeConvolutionBlock'   : RawSizeConvolutionBlock,
        'CroppedConvolutionBlock'   : CroppedConvolutionBlock,
        'SegmentaryConvolutionBlock': SegmentaryConvolutionBlock,
        'CustomizedConvolutionModel': CustomizedConvolutionModel
    }
    model = load_weights(model_path)

    print_model(model)