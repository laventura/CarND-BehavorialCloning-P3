
from __future__ import print_function

import numpy as np 
import pandas as pd 
import time
import os, sys
import cv2

import argparse

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Convolution2D, MaxPooling2D, ZeroPadding2D, Dense, Activation, ELU
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.layers import Lambda
from keras.preprocessing.image import img_to_array, load_img, flip_axis, random_shift
from keras.models import model_from_json

from sklearn.model_selection import train_test_split

from collections import Counter
from sklearn.utils import shuffle

# Udacity driving data
DATA_DIR = '/Users/aa/Developer/courses/self_driving_carnd/P3_behavior_cloning/data/udacity-data'
DATA_DIR6 = '/Users/aa/Downloads/drive6'
#DATA_DIR = '/Users/aa/Downloads/drive10'   # Center image only

DRIVING_LOG = 'driving_log.csv'

BRIGHTNESS_RANGE = 0.2
X_RANGE_TRANS = 50
Y_RANGE_TRANS = 30
ANGLE_TRANS = 0.3

BATCH_SIZE = 128
IMG_HEIGHT, IMG_WIDTH = 66, 200    # Nvidia model
# IMG_HEIGHT, IMG_WIDTH = 64, 64    # VGG model
NUM_CHANNELS = 3

STEERING_BIAS = 0.3

NB_EPOCH=5

# dtypes for Driving Log file columns
coltypes = {'center': str,
         'left': str,
         'right': str,
         'steering': np.float64,
         'throttle': np.float64,
         'brake': np.float64,
         'speed': np.float64}


def augment_brightness(image):
    ''' Add random brightness to given image (RGB) by converting to HSV 
        and converting back to RGB
        :param image - the given image
        :return image
    '''
    image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    random_bright = np.random.uniform() + BRIGHTNESS_RANGE
    image[:, :, 2] = image[:, :, 2] * random_bright
    image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
    return image

def preprocess_image(image):
    ''' 
        Pre-process the given image
        Remove top 60 pixels (y); and for width keep only 250 pixels
        :param image - input image of size 160 (H) x 320 (W)
        :return the processed image
    '''
    # ignore the top part of image
    image = image[60:140, 30:280, :]   # Height: ignore top 60 pix; Width: take only 30-280 pix
    # Resize the image
    image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_AREA)
    return np.resize(image, (1, IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS))


def translate_image(image, tx):
    '''
        Translate the image in X direction, specified by the translation matrix tx
        :param image - input 
        :param tx - the translation matrix
        :return - the translated image
    '''
    # random y translation
    ty = (Y_RANGE_TRANS * np.random.uniform()) - (Y_RANGE_TRANS / 2)

    translation_matrix = np.float32([[1, 0, tx], 
                                    [0, 1, ty]])
    # translate 
    return cv2.warpAffine(image, translation_matrix, (image.shape[1], image.shape[0]))

def augment_image(img_file, steering_angle):
    ''' Augment the given image randomly. Add brightness, translate, and/or randomly flip the image, 
        and pre-process it
        :param img_file - input image 
        :param steering_angle - the augmented steering angle 
        :return (processed image, steering angle)
    '''
    # print('**** CV2 opening: ', img_file)
    image = cv2.imread(img_file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 1. augment brightness
    image = augment_brightness(image)

    # 2. random translate 
    tx = (X_RANGE_TRANS * np.random.uniform()) - (X_RANGE_TRANS / 2)
    angle2 = steering_angle + ((tx / X_RANGE_TRANS) * 2) * ANGLE_TRANS

    ## TODO:  Check if angle2 exceeds a threshold???
    image = translate_image(image, tx)

    # 3. randomly flip 
    if np.random.randint(2) == 0:
        image = cv2.flip(image, 1)
        steering_angle = -steering_angle

    # 4. pre preprocess 
    image = preprocess_image(image)
    return image, steering_angle


def test_df(df, BASE_DIR=DATA_DIR):
    ''' Debug function to test images from the given dataframe
    '''
    ix = np.random.randint(len(df))
    print('ix:', ix)
    angle = float(df['steering'].iloc[ix])
    print('angle:', angle)

    L = df.left.iloc[ix].strip()
    R = df.right.iloc[ix].strip()
    C = df.center.iloc[ix].strip()

    if L[0] != '/':
        L = os.path.join(BASE_DIR, L)
    if R[0] != '/':
        R = os.path.join(BASE_DIR, R)
    if C[0] != '/':
        C = os.path.join(BASE_DIR, C)
    
    print('C:', C)
    print('R:', R)
    print('L:', L)
    Limg = cv2.imread(L)
    print('L shape:', Limg.shape)


def train_data_generator(df, BASE_DIR=DATA_DIR):
    ''' Continuously generate training data, given df data frame with all the driving info. 
        Generator function to be used with Keras model's fit_generator()
        Returns batches of X (training data images), y (steering angle)

        :param df  dataframe with all driving info
        :return  (X[BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS], y)
    '''
    _x  = np.zeros((BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS), dtype=np.float)
    _y = np.zeros(BATCH_SIZE, dtype=np.float)
    output_idx = 0

    while True:
        ix = np.random.randint(len(df))
        steering_angle = float(df.steering.iloc[ix])

        # select a random image from CENTER, LEFT, RIGHT
        rand = np.random.randint(3)

        if rand == 0:  # CENTER
            img_file = df.center.iloc[ix].strip()
        elif rand == 1:  # LEFT
            img_file = df.left.iloc[ix].strip()
            steering_angle += STEERING_BIAS
        else:  # RIGHT
            img_file = df.right.iloc[ix].strip()
            steering_angle -= STEERING_BIAS
 
        # if image is NaN or nan - ignore and move on 
        # some simulator driving logs have no Right/Left images
        if img_file == '' or img_file == 'nan' or img_file == 'NaN':
            continue
       
        # ensure no NaN or nan or blank
        if img_file != '' and img_file != 'nan' and img_file != 'NaN' and img_file[0] != '/':
            img_file.strip()
            img_file = os.path.join(BASE_DIR, img_file)

        # randomly augment the image
        img_file.strip()
        image, steering_angle = augment_image(img_file, steering_angle)

        if image is not None:
            _x[output_idx] = image
            _y[output_idx] = steering_angle
            output_idx += 1

        if output_idx >= BATCH_SIZE:
            yield _x, _y

            # reset
            _x  = np.zeros((BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS), dtype=np.float)
            _y = np.zeros(BATCH_SIZE, dtype=np.float)
            output_idx = 0


def validation_generator(df, BASE_DIR=DATA_DIR):
    ''' Generate validation data (continuously) using given df dataframe
        Generates batches of X (validation image), y (steering angle) 
        :param df  dataframe with all driving info
        :return  (X[BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS], y)
    '''
    # assert len(df) == BATCH_SIZE, 'Length of validation df should be BATCH_SIZE'

    while True:
        x = np.zeros((BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS), dtype=np.float)
        y = np.zeros(BATCH_SIZE, dtype=np.float)

        for ix in np.arange(BATCH_SIZE):
            rand =  np.random.randint(len(df))
            center_img_file = df.center.iloc[rand].strip()
            if center_img_file[0] != '/':
                center_img_file = os.path.join(BASE_DIR, center_img_file)
            x[ix] = preprocess_image(cv2.imread(center_img_file))
            y[ix] = df.steering.iloc[rand]

        yield x, y


def test_predictions(model, df, tries=5, BASE_DIR=DATA_DIR):
    ''' 
        Test the given model on a dataframe of test data
    '''
    print('** predictions **')
    for i in np.arange(tries):
        dfset = df.loc[df.steering  < (i * 0.4) - 0.6]
        subset = dfset.loc[dfset.steering >= (i * 0.4) - 1.]
        ix = int(len(subset)/2)
        center_img = subset.center.iloc[ix].strip()
        if center_img[0] != '/':
            center_img = os.path.join(BASE_DIR, center_img)
        img = preprocess_image(cv2.imread(center_img))
        img = np.resize(img, (1, IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS))
        true_angle = subset.steering.iloc[ix]
        pred_angle = model.predict(img, batch_size=1)
        print(true_angle, pred_angle[0][0])
    print()



def create_nvidia_model(shape=(IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS)):
    ''' Create and return a Keras model (based on Nvidia's end-to-end driving)
        :param shape - input shape (currently 66(H)x200(W)x3(Channels)
        :return the Keras model
    '''
    
    print('Creating Model Nvidia for shape:', shape)
    model = Sequential()
    # normalize between -0.5to 0.5
    model.add(Lambda(lambda x: x/255. - 0.5, input_shape=shape, output_shape=shape))
    
    # Conv layers
    model.add(Convolution2D(24, 5, 5, border_mode='valid', subsample=(2, 2), 
                            activation='elu', name='conv0'))
    #model.add(Dropout(0.2))
    model.add(Convolution2D(36, 5, 5, border_mode='valid', subsample=(2, 2), 
                           activation='elu', name='conv1'))
    model.add(Dropout(0.2))
    model.add(Convolution2D(48, 5, 5, border_mode='valid', subsample=(2, 2),
                           activation='elu', name='conv2'))
    #model.add(Dropout(0.2))
    model.add(Convolution2D(64, 3, 3, border_mode='valid', subsample=(1, 1),
                           activation='elu', name='conv3'))
    model.add(Dropout(0.2))
    model.add(Convolution2D(64, 3, 3, border_mode='valid', subsample=(1, 1),
                           activation='elu', name='conv4'))
    #model.add(Dropout(0.2))
    
    model.add(Flatten())
    #model.add(Dropout(0.5))
    
    # FC layers
    model.add(Dense(1024, activation='elu',name='fc0'))
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='elu',name='fc1'))
    model.add(Dropout(0.5))
    #model.add(Dense(50, activation='elu',name='fc2'))
    #model.add(Dropout(0.5))
    model.add(Dense(10, activation='elu',name='fc3'))
    model.add(Dropout(0.3))
    model.add(Dense(1, name='fc4'))
    
    print(model.summary())

    # model.compile(loss='mse', optimizer='adam')
    return model

def create_vgg_model(shape=(IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS)):
    ''' Create a VGG model 
    '''
    print('Creating Model VGG for shape:', shape)
    model = Sequential()
    # normalize between -0.5to 0.5
    model.add(Lambda(lambda x: x/255. - 0.5, input_shape=shape, output_shape=shape))

    model.add(Convolution2D(64, 3, 3, activation='elu', border_mode='same', name='bk1_conv1'))
    model.add(Convolution2D(64, 3, 3, activation='elu', border_mode='same', name='bk1_conv2'))
    model.add(MaxPooling2D((2,2), strides=(2,2), name='bk1_pool'))

    model.add(Convolution2D(128, 3, 3, activation='elu', border_mode='same', name='bk2_conv1'))
    model.add(Convolution2D(128, 3, 3, activation='elu', border_mode='same', name='bk2_conv2'))
    model.add(MaxPooling2D((2,2), strides=(2,2), name='bk2_pool'))

    model.add(Dropout(0.5))

    model.add(Convolution2D(256, 3, 3, activation='elu', border_mode='same', name='bk3_conv1'))
    model.add(Convolution2D(256, 3, 3, activation='elu', border_mode='same', name='bk3_conv2'))
    model.add(MaxPooling2D((2,2), strides=(2,2), name='bk3_pool'))

    #model.add(Convolution2D(512, 3, 3, activation='elu', border_mode='same', name='bk4_conv1'))
    #model.add(Convolution2D(512, 3, 3, activation='elu', border_mode='same', name='bk4_conv2'))
    #model.add(MaxPooling2D((2,2), strides=(2,2), name='bk4_pool'))

    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='elu', name='fc1'))
    model.add(Dropout(0.5))
    # model.add(Dense(512, activation='elu', name='fc2'))
    # model.add(Dropout(0.5))
    model.add(Dense(256, activation='elu', name='fc3'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='elu', name='fc4'))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='elu', name='fc5'))
    model.add(Dropout(0.5))

    model.add(Dense(1, init='zero', name='final'))

    print(model.summary())
    return model

def create_small_vgg(shape=(IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS)):
    print('Creating small vgg model for shape:', shape)
    model = Sequential()
    # normalize between -0.5to 0.5
    model.add(Lambda(lambda x: x/255. - 0.5, input_shape=shape, output_shape=shape))

    model.add(Convolution2D(32, 3, 3, activation='elu', border_mode='same', name='bk1_conv1'))
    model.add(Convolution2D(32, 3, 3, activation='elu', border_mode='same', name='bk1_conv2'))
    model.add(MaxPooling2D((2,2), strides=(2,2), name='bk1_pool'))

    model.add(Convolution2D(64, 3, 3, activation='elu', border_mode='same', name='bk2_conv1'))
    model.add(Convolution2D(164, 3, 3, activation='elu', border_mode='same', name='bk2_conv2'))
    model.add(MaxPooling2D((2,2), strides=(2,2), name='bk2_pool'))
    model.add(Dropout(0.5))

    model.add(Convolution2D(128, 3, 3, activation='elu', border_mode='same', name='bk3_conv1'))
    model.add(Convolution2D(128, 3, 3, activation='elu', border_mode='same', name='bk3_conv2'))
    model.add(MaxPooling2D((2,2), strides=(2,2), name='bk3_pool'))

    model.add(Dense(256, activation='elu', name='fc1'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='elu', name='fc2'))
    model.add(Dropout(0.5))
    model.add(Dense(16, activation='elu', name='fc3'))

    model.add(Dense(1, name='final'))

    print(model.summary())
    return model


def load_driving_data(driving_log_file, BASE_DIR=DATA_DIR):
    '''  Load driving data from driving log file; add BASE_DIR if not absolute filename
        :param  driving_log_file - log file with driving data 
        return: pandas dataframe
    '''
    print('** From:', BASE_DIR, ' Reading file:', driving_log_file)
    df = pd.read_csv(os.path.join(BASE_DIR,driving_log_file), 
                     names=['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed'],
                     dtype=coltypes, header=0)
    df = df[1:]  # ignore the first line (header)

    # some driving log files do not have Right/Left images; take care of these NaN
    df['left']  = df['left'].apply(lambda x: str(x))  # convert all 'left to str
    df['right'] = df['right'].apply(lambda x: str(x))  # convert all 'left to str
    y_train = df['steering'].values
    print('Found samples:', len(df))
    return df, y_train

def train(train_data, val_data, model_name='model', weights=None, load=False):
    '''  Run training and validation on model 
        :param train_data - data frame with training data
        :param val_data  - data frame with validation data
        :param model_name - name of model to save
        :param weights - (optional) name of weights file to load
        :param load - load weights flag
    '''

    model = create_nvidia_model()
    # model = create_vgg_model()
    # model = create_small_vgg()

    if load and weights is not None and os.path.isfile(weights):
        print('### loading weights.. )', weights)
        model.load_weights(weights)

    # callbacks - save best weight
    ckpoint_path=model_name+"-weights-{epoch:02d}-{val_loss:.2f}.hdf5"
    model_checkpoint = ModelCheckpoint(ckpoint_path, monitor='val_loss', verbose=0, \
    save_best_only=True, save_weights_only=False, mode='auto')

    # Learning Rate Schedule: reduce LR when val_loss does not decrease for patience number of epochs
    lr_schedule = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, verbose=1)

    # optimizer
    opt = Adam(lr=0.0001)
    model.compile(loss='mse', optimizer=opt)

    # test some predictions
    test_predictions(model, val_data, BASE_DIR=DATA_DIR)

    with open(model_name+'.json', 'w') as fout:
        fout.write(model.to_json())
        print('saved json model to [.json]', model_name)

    # run training
    history = model.fit_generator(train_data_generator(train_data, BASE_DIR=DATA_DIR),
                                samples_per_epoch=BATCH_SIZE * (len(train_data) // BATCH_SIZE),  
                                nb_epoch=NB_EPOCH,
                                validation_data=validation_generator(val_data, BASE_DIR=DATA_DIR),
                                nb_val_samples=BATCH_SIZE,
                                callbacks=[model_checkpoint, lr_schedule],
                                verbose=1)


    # save model 
    model.save(model_name+'.h5')
    #model.save_weights(model_name+'.h5')
    print('saved model+weights to (.h5)', model_name)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Training Driving')
    parser.add_argument('model', type=str, help='Name of model to create [default: model.json].')
    parser.add_argument('-weights', type=str, help='Weights of the model to load.')

    args = parser.parse_args()
    model = args.model
    print('*** Creating model: ', model)

    data_all, y_train = load_driving_data(DRIVING_LOG, BASE_DIR=DATA_DIR)

    # load additional driving data 
    additional_data, _ = load_driving_data(DRIVING_LOG, BASE_DIR=DATA_DIR6 )
    # combine
    data_all = data_all.append(additional_data)

    print('Total data len     :', len(data_all))

    # split
    val_data, train_data = np.split(data_all.sample(frac=1), [BATCH_SIZE * 5])
    print('Validation data len:', len(val_data))
    print('Train data len     :', len(train_data))

    # run training
    train(train_data, val_data, model_name=model, weights=args.weights, load=True)


