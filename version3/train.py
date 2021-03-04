''' Parameters '''
MODEL = 'ClassicalConvolutionModel'
# MODEL = 'CustomizedConvolutionModel'
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
from tensorflow.keras.models import load_model

from models.ClassicalConvolutionModel import ClassicalConvolutionModel, LastLayer
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

    renamed_save_directory = f"{save_directory}-{min(history['val_loss']):.4f}"
    os.rename(save_directory, renamed_save_directory)

    plt.savefig(f'{renamed_save_directory}/comparition_figure.png', dpi=200)
    # plt.show()

    print('Done.')
    return renamed_save_directory


def read_train_data():
    print('')
    X = []
    for i in tqdm(range(1, 1425), desc=f"Reading training images", ascii=True):
        if i in TRAIN_IGNORING: continue
        original_im = np.array(Image.open(f"processed_train_data/image_{i:04d}_original.png"))
        color_im    = np.array(Image.open(f"processed_train_data/image_{i:04d}_color.png"))
        cropped_im  = np.array(Image.open(f"processed_train_data/image_{i:04d}_cropped.png"))
        if MODEL == 'ClassicalConvolutionModel':
            X.append(np.concatenate([
                original_im, color_im, cropped_im,
            ], axis=2))
        elif MODEL == 'CustomizedConvolutionModel':
            X.append([ original_im, color_im, cropped_im ])
    X = np.array(X)
    print('\nReading training annotations... ', end='')
    Y = pd.read_csv(r"processed_train_data/annotation.csv", index_col=0).values
    Y = np.array([ answer for index, answer in enumerate(Y, start=1) if index not in TRAIN_IGNORING ])
    print('Done.\n')
    return X, Y


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


def euclidean_distance_loss(y_true, y_pred):
    y_true = tf.Variable(np.array([ [ [y[0], y[1]], [y[2], y[3]] ] for y in y_true ]), dtype=tf.float32)
    y_pred = tf.Variable(np.array([ [ [y[0], y[1]], [y[2], y[3]] ] for y in y_pred ]), dtype=tf.float32)
    return tf.reduce_sum(tf.norm(y_true - y_pred, axis=1, ord='euclidean'), axis=-1)
    
    
def my_euclidean_distance_loss(y_true, y_pred):
    y_true = tf.dtypes.cast(y_true, tf.float32)
    point_1 = ((y_true[:, 0] - y_pred[:, 0])**2 + (y_true[:, 1] - y_pred[:, 1])**2)**0.5
    point_2 = ((y_true[:, 2] - y_pred[:, 2])**2 + (y_true[:, 3] - y_pred[:, 3])**2)**0.5
    return (point_1 + point_2)


''' Execution '''
if __name__ == '__main__':

    # Clear screen
    os.system('cls')
    
    # Build model
    if MODEL == 'ClassicalConvolutionModel':
        model = ClassicalConvolutionModel(image_amount=3)
        opt = tf.keras.optimizers.Adam(learning_rate=0.01)
        model.compile(loss=my_euclidean_distance_loss, optimizer=opt, run_eagerly=True)
        print('')
        model.summary()

    elif MODEL == 'CustomizedConvolutionModel':
        model = CustomizedConvolutionModel()
        opt = tf.keras.optimizers.Adam(learning_rate=0.01)
        model.compile(loss=my_euclidean_distance_loss, optimizer=opt, run_eagerly=True)

    # Read training data
    X_train, Y_train = read_train_data()

    # Set saving path
    now = time.localtime()
    date = time.strftime('%Y-%m-%d', now)
    time = time.strftime('%H.%M.%S', now)
    if   MODEL == 'ClassicalConvolutionModel' : save_directory = f'../output/{date}/{time}-ver3-classical'
    elif MODEL == 'CustomizedConvolutionModel': save_directory = f'../output/{date}/{time}-ver3-customized'
    if not os.path.exists(save_directory): os.makedirs(save_directory)

    # Set EarlyStopping & ModelCheckpoint
    ES = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        mode='min',
        verbose=True,
        patience=20
    )
    # MCP_loss = tf.keras.callbacks.ModelCheckpoint(
    #     filepath=f'{save_directory}/best_model_min_loss.h5',
    #     monitor='loss',
    #     verbose=True,
    #     mode='min',
    #     save_best_only=True
    # )
    MCP_val_loss = tf.keras.callbacks.ModelCheckpoint(
        filepath=f'{save_directory}/best_model_min_val_loss.h5',
        monitor='val_loss',
        verbose=True,
        mode='min',
        save_best_only=True
    )

    # Train model
    # 1424 - 30 = 1394  -->  1394 * 0.75 = 1045
    history = model.fit(
        X_train, Y_train,
        batch_size=19,
        epochs=500,
        callbacks=[ ES, MCP_val_loss ],
        # callbacks=[ ES, MCP_loss, MCP_val_loss ],
        validation_split=0.25,
        shuffle=True,
    )

    # Save comparition figure
    save_directory = save_comparition_figure(history.history, save_directory)

    # Predict answer
    X_test = read_test_data()
    custom_objects = {'my_euclidean_distance_loss': my_euclidean_distance_loss, 'LastLayer': LastLayer}
    model = load_model(f'{save_directory}/best_model_min_val_loss.h5', custom_objects=custom_objects)
    print('Pedicting...')
    Y_pred = model.predict(X_test, verbose=True)
    image_names = np.array([ [f'image_{i+1:04d}.png'] for i in range(1000) ])
    Y_pred = np.concatenate((image_names, Y_pred), axis=1)
    Y_pred = pd.DataFrame(Y_pred, columns=['images', 'x1', 'y1', 'x2', 'y2'])
    Y_pred.to_csv(f'{save_directory}/submission.csv', index=False)
    print('Done.')