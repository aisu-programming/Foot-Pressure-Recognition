''' Libraries '''
import os
from tqdm import tqdm
from PIL import Image
import numpy as np
import pandas as pd

from models.mingleConvoluntion import mingleConvoluntion


''' Parameters '''
train_ignoring   = [ 67,  83, 170, 171, 235, 245, 313, 373, 404, 410, 478, 589, 600, 609, 625, 637, 649, 656, 662, 685, 749, 751, 797, 801, 826, 872, 891, 933, 953, 980]
test_malfunction = [ 91,  95, 111, 153]


''' Functions '''
def read_data():

    X = []
    for i in tqdm(range(1, 1425, 1), desc=f"Reading images", ascii=True):
        if i in train_ignoring: continue
        cropped_image                  = np.array(Image.open(f"test/1/image_{i:04d}_cropped.png"))
        edges_image                    = np.array(Image.open(f"test/1/image_{i:04d}_edges.png"))
        detail_image                   = np.array(Image.open(f"test/1/image_{i:04d}_detail.png"))
        sharpen_image                  = np.array(Image.open(f"test/1/image_{i:04d}_sharpen.png"))
        contour_image                  = np.array(Image.open(f"test/1/image_{i:04d}_contour.png"))
        contour_sharpen_image          = np.array(Image.open(f"test/1/image_{i:04d}_contour_sharpen.png"))
        adjusted_contour_sharpen_image = np.array(Image.open(f"test/1/image_{i:04d}_adjusted_contour_sharpen.png"))
        input_images = [
            cropped_image, edges_image, detail_image, sharpen_image, contour_image,
            contour_sharpen_image, adjusted_contour_sharpen_image,
        ]
        mingle_images = np.empty((316, 102, 0))
        for array in input_images:
            mingle_images = np.concatenate([mingle_images, array], axis=2)
        X.append(mingle_images)
    X = np.array(X)

    print('Reading annotations.')
    Y = pd.read_csv(r"test/1/annotation.csv", index_col=0).values
    Y = np.array([ answer for index, answer in enumerate(Y, start=1) if index not in train_ignoring ])

    return X, Y


''' Execution '''
if __name__ == '__main__':
    os.system('cls')
    X, Y = read_data()
    model = mingleConvoluntion(image_amount=7)
    model.fit(X, Y, batch_size=40, epochs=500, validation_split=0.3)