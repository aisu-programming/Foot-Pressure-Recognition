''' Libraries '''
import os
from tqdm import tqdm
from PIL import Image
import numpy as np
import pandas as pd

from models.ClassicalConvoluntionModel import ClassicalConvoluntionModel


''' Parameters '''
MODEL = 'ClassicalConvolutionModel'
# MODEL = 'CustomizedConvolutionModel'
TRAIN_IGNORING   = [ 67,  83, 170, 171, 235, 245, 313, 373, 404, 410, 478, 589, 600, 609, 625, 637, 649, 656, 662, 685, 749, 751, 797, 801, 826, 872, 891, 933, 953, 980]
TEST_MALFUNCTION = [ 91,  95, 111, 153]


''' Functions '''
def read_data():
    print('')

    X = []
    for i in tqdm(range(1, 1425, 1), desc=f"Reading images", ascii=True):
        if i in TRAIN_IGNORING: continue

        original_image                 = np.array(Image.open(f"test/1/image_{i:04d}_original.png"))
        color_image                    = np.array(Image.open(f"test/1/image_{i:04d}_color.png"))
        cropped_image                  = np.array(Image.open(f"test/1/image_{i:04d}_cropped.png"))
        edges_image                    = np.array(Image.open(f"test/1/image_{i:04d}_edges.png"))
        # detail_image                   = np.array(Image.open(f"test/1/image_{i:04d}_detail.png"))
        sharpen_image                  = np.array(Image.open(f"test/1/image_{i:04d}_sharpen.png"))
        contour_image                  = np.array(Image.open(f"test/1/image_{i:04d}_contour.png"))
        contour_sharpen_image          = np.array(Image.open(f"test/1/image_{i:04d}_contour_sharpen.png"))
        # adjusted_contour_sharpen_image = np.array(Image.open(f"test/1/image_{i:04d}_adjusted_contour_sharpen.png"))

        if MODEL == 'ClassicalConvolutionModel':
            X.append(
                np.concatenate([
                    cropped_image, edges_image, sharpen_image, contour_image, contour_sharpen_image,
                ], axis=2)
            )
        elif MODEL == 'CustomizedConvolutionModel':
            X.append({
                'original'              : original_image,
                'color'                 : color_image,
                'cropped'               : cropped_image,
                'cropped_pieces'        : cropped_image,
                'edges'                 : edges_image,
                'edges_pieces'          : edges_image,
                'sharpen'               : sharpen_image,
                'sharpen_pieces'        : sharpen_image,
                'contour'               : contour_image,
                'contour_pieces'        : contour_image,
                'contour_sharpen'       : contour_sharpen_image,
                'contour_sharpen_pieces': contour_sharpen_image,
            })

    X = np.array(X)

    print('Reading annotations.')
    Y = pd.read_csv(r"test/1/annotation.csv", index_col=0).values
    Y = np.array([ answer for index, answer in enumerate(Y, start=1) if index not in TRAIN_IGNORING ])

    print('')
    return X, Y


''' Execution '''
if __name__ == '__main__':
    os.system('cls')
    
    model = ClassicalConvoluntionModel(image_amount=5)
    
    X, Y = read_data()

    # ES = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
    model.fit(X, Y, batch_size=1, epochs=500, validation_split=0.3)