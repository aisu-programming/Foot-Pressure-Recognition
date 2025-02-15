''' Parameters '''
# MODEL = 'ClassicalConvolutionModel'
MODEL = 'CustomizedConvolutionModel'
TRAIN_IGNORING   = [
     67,  83, 170, 171, 235, 245, 313, 373, 404, 410,
    478, 589, 600, 609, 625, 637, 649, 656, 662, 685,
    749, 751, 797, 801, 826, 872, 891, 933, 953, 980
]
TEST_MALFUNCTION = [ 91,  95, 111, 153]
RAMDON_SEED = 1


''' Libraries '''
# Programming
import os
import time
from tqdm import tqdm
import matplotlib.pyplot as plt

# Data processing
from PIL import Image
import numpy as np
import pandas as pd

np.random.seed(RAMDON_SEED)

# Deep learning
import tensorflow as tf

from models.ClassicalConvolutionModel import ClassicalConvolutionModel
from models.CustomizedConvolutionModel import CustomizedConvolutionModel


''' Functions '''
def save_comparition_figure(history, save_directory):

    if not os.path.exists(save_directory): os.makedirs(save_directory)

    loss = history['loss']
    validation_loss = history['val_loss']
    epochs_length = range(1, len(loss)+1)

    fig, ax = plt.subplots()
    fig.set_size_inches(12, 16) # 3:4
    ax.plot(epochs_length, loss, "b-", label='Training Loss')
    ax.plot(epochs_length, validation_loss, "r-", label='Validation Loss')
    ax.set(xlabel='Epochs', ylabel='Loss', title='Training & Validation Comparition')
    ax.grid()

    plt.savefig(f'{save_directory}/comparition_figure.png', dpi=200)
    # plt.show()

    return


def read_data():
    print('')

    X = []
    for i in tqdm(range(1, 1425), desc=f"Reading images", ascii=True):
        if i in TRAIN_IGNORING: continue
        original_im = np.array(Image.open(f"processed_data/image_{i:04d}_original.png"))
        color_im    = np.array(Image.open(f"processed_data/image_{i:04d}_color.png"))
        cropped_im = np.array(Image.open(f"processed_data/image_{i:04d}_cropped.png"))
        if MODEL == 'ClassicalConvolutionModel':
            X.append(np.concatenate([
                original_im, color_im, cropped_im,
            ], axis=2))
        elif MODEL == 'CustomizedConvolutionModel':
            X.append([ original_im, color_im, cropped_im ])
    X = np.array(X)

    print('Reading annotations.')
    Y = pd.read_csv(r"processed_data/annotation.csv", index_col=0).values
    Y = np.array([ answer for index, answer in enumerate(Y, start=1) if index not in TRAIN_IGNORING ])

    print('')
    return X, Y


''' Execution '''
if __name__ == '__main__':
    os.system('cls')
    
    if MODEL == 'ClassicalConvolutionModel':
        model = ClassicalConvolutionModel(image_amount=3)
        model.compile(loss="mean_absolute_percentage_error", optimizer='adam')

    elif MODEL == 'CustomizedConvolutionModel':
        model = CustomizedConvolutionModel()
        opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
        model.compile(loss="mean_absolute_percentage_error", optimizer=opt, run_eagerly=True)
    
    X, Y = read_data()

    ES = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
    # 1424 - 30 = 1394  -->  1394 * 0.75 = 1045
    history = model.fit(X, Y, batch_size=11, epochs=100, callbacks=[ES], validation_split=0.25, shuffle=True)
    now = time.localtime()
    date = time.strftime('%Y-%m-%d', now)
    time = time.strftime('%H.%M.%S', now)
    save_comparition_figure(history.history, f'../output/{date}/{time} (version 2)')