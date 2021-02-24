''' Libraries '''
import os
import math
from tqdm import tqdm
from PIL import Image
import numpy as np
import pandas as pd

from models.ClassicalConvolutionModel import ClassicalConvolutionModel


''' Parameters '''
# MODEL = 'ClassicalConvolutionModel'
MODEL = 'CustomizedConvolutionModel'
TRAIN_IGNORING   = [ 67,  83, 170, 171, 235, 245, 313, 373, 404, 410, 478, 589, 600, 609, 625, 637, 649, 656, 662, 685, 749, 751, 797, 801, 826, 872, 891, 933, 953, 980]
TEST_MALFUNCTION = [ 91,  95, 111, 153]


''' Functions '''
def segment_image(image):
    segmentations = []
    for w in range(6):
        for h in range(19):
            left   = math.floor(102 * (w / 6.0))
            right  = math.ceil(102 * ((w+1) / 6.0))
            top    = math.floor((316+7) * (h / 19.0))
            bottom = top + 10
            segmentations.append(image[top:bottom, left:right, :].shape)
    return np.array(segmentations)


def read_data():
    print('')

    X = []
    for i in tqdm(range(1, 1425, 1), desc=f"Reading images", ascii=True):
        if i in TRAIN_IGNORING: continue

        original_im                 = np.array(Image.open(f"test/1/image_{i:04d}_original.png"))
        color_im                    = np.array(Image.open(f"test/1/image_{i:04d}_color.png"))
        cropped_im                  = np.array(Image.open(f"test/1/image_{i:04d}_cropped.png"))
        edges_im                    = np.array(Image.open(f"test/1/image_{i:04d}_edges.png"))
        # detail_im                   = np.array(Image.open(f"test/1/image_{i:04d}_detail.png"))
        sharpen_im                  = np.array(Image.open(f"test/1/image_{i:04d}_sharpen.png"))
        contour_im                  = np.array(Image.open(f"test/1/image_{i:04d}_contour.png"))
        contour_sharpen_im          = np.array(Image.open(f"test/1/image_{i:04d}_contour_sharpen.png"))
        # adjusted_contour_sharpen_im = np.array(Image.open(f"test/1/image_{i:04d}_adjusted_contour_sharpen.png"))

        if MODEL == 'ClassicalConvolutionModel':
            X.append(np.concatenate([
                cropped_image, edges_image, sharpen_image, contour_image, contour_sharpen_image,
            ], axis=2))
        elif MODEL == 'CustomizedConvolutionModel':
            X.append({
                # OverallConvolution (raw_size)
                'original'              : original_im,
                'color'                 : color_im,
                # OverallConvolution
                'cropped'               : cropped_im,
                'edges'                 : edges_im,
                'sharpen'               : sharpen_im,
                'contour'               : contour_im,
                'contour_sharpen'       : contour_sharpen_im,
                # SegmentaryConvolution
                'cropped_pieces'        : segment_image(cropped_im),  
                'edges_pieces'          : segment_image(edges_im),
                'sharpen_pieces'        : segment_image(sharpen_im),
                'contour_pieces'        : segment_image(contour_im),
                'contour_sharpen_pieces': segment_image(contour_sharpen_im),
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
    
    # model = ClassicalConvolutionModel(image_amount=5)
    
    X, Y = read_data()

    # ES = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
    model.fit(X, Y, batch_size=1, epochs=500, validation_split=0.3)