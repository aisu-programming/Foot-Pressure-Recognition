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

    print(f"\nSaving comparition figure... ", end='')

    loss = history['loss']
    validation_loss = history['val_loss']
    epochs_length = range(1, len(loss)+1)

    fig, ax = plt.subplots()
    fig.set_size_inches(5, 4)
    ax.plot(epochs_length, loss, "b-", label='Training Loss')
    ax.plot(epochs_length, validation_loss, "r-", label='Validation Loss')
    ax.set(xlabel='Epochs', ylabel='Loss', title='Training & Validation Comparition')
    ax.grid()

    os.rename(save_directory, f"{save_directory}-{min(history['val_loss'])}")

    plt.savefig(f'{save_directory}/comparition_figure.png', dpi=200)
    # plt.show()

    print('Done.')
    return


# def record_details(cost_time, test_loss, test_accuracy, original_loss, original_accuracy, model_scores):
#     print(f"\nRecording details... ", end='')
#     with open(f'{save_directory}/details.txt', mode='w') as f:
#         f.write(f'RAMDON_SEED = {RAMDON_SEED}\n')
#         f.write(f'\n')
#         f.write(f'data_divide_amount = {data_divide_amount}\n')
#         f.write(f'frames_per_data = {frames_per_data}\n')
#         f.write(f'\n')
#         f.write(f'test_split = {test_split}\n')
#         f.write(f'validation_split = {validation_split}\n')
#         # f.write(f'train data proportion: {test_split * (1 - validation_split):2.0f}%\n')
#         # f.write(f'test data proportion: {(1 - test_split) * 100:2.0f}%\n')
#         # f.write(f'validation data proportion: {test_split * validation_split:2.0f}%\n')
#         f.write(f'\n')
#         f.write(f'epochs = {epochs}\n')
#         f.write(f'batch_size = {batch_size}\n')
#         f.write(f'\n')
#         f.write(f'cost_time: {cost_time}\n')
#         f.write(f'test_loss: {test_loss}\n')
#         f.write(f'test_accuracy: {test_accuracy}\n')
#         f.write(f'original_loss (all data): {original_loss}\n')
#         f.write(f'original_accuracy (all data): {original_accuracy}\n')
#         f.write(f'\n')
#         for model_name, model_score in model_scores.items():
#             f.write(f"Score by '{model_name}' model: {model_score}\n")
#     print('Done.')
#     return


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

    # Clear screen
    os.system('cls')

    # Set saving path
    now = time.localtime()
    date = time.strftime('%Y-%m-%d', now)
    time = time.strftime('%H.%M.%S', now)
    save_directory = f'../output/{date}/{time}-ver3'
    if not os.path.exists(save_directory): os.makedirs(save_directory)
    
    # Build model
    if MODEL == 'ClassicalConvolutionModel':
        model = ClassicalConvolutionModel(image_amount=3)
        model.compile(loss="mean_absolute_percentage_error", optimizer='adam')

    elif MODEL == 'CustomizedConvolutionModel':
        model = CustomizedConvolutionModel()
        opt = tf.keras.optimizers.Adam(learning_rate=0.001)
        model.compile(loss="mean_absolute_percentage_error", optimizer=opt, run_eagerly=True)
    
    # Read training data
    X, Y = read_data()

    # Set EarlyStopping & ModelCheckpoint
    ES = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        mode='min',
        verbose=True,
        patience=10
    )
    MCP_loss = tf.keras.callbacks.ModelCheckpoint(
        filepath=f'{save_directory}/best_model_min_loss.h5',
        monitor='loss',
        mode='min',
        save_best_only=True
    )
    MCP_val_loss = tf.keras.callbacks.ModelCheckpoint(
        filepath=f'{save_directory}/best_model_min_val_loss.h5',
        monitor='val_loss',
        mode='min',
        save_best_only=True
    )

    # Train model
    # 1424 - 30 = 1394  -->  1394 * 0.75 = 1045
    history = model.fit(
        X, Y,
        batch_size=19,
        epochs=100,
        callbacks=[ ES, MCP_loss, MCP_val_loss ],
        validation_split=0.25,
        shuffle=True
    )

    # Save comparition figure
    save_comparition_figure(history.history, save_directory)