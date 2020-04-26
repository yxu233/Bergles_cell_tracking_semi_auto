# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 22:18:01 2020

@author: Huganir Lab
"""

import bcolz
import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt
plt.style.use("ggplot")


from keras.models import Model, load_model
from keras.layers import Input, BatchNormalization, Activation, Dense, Dropout
from keras.layers.core import Lambda, RepeatVector, Reshape
from keras.layers.convolutional import Conv2D, Conv2DTranspose, Conv3D, Conv3DTranspose
from keras.layers.pooling import MaxPooling2D, GlobalMaxPool2D, MaxPooling3D, GlobalMaxPooling3D
from keras.layers.merge import concatenate, add
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

from sklearn.model_selection import train_test_split

sav_dir = '../Train_tracking_data/Train_tracking_data_analytic_results/'

X = bcolz.open(sav_dir + 'input_im', mode='r')
Y = bcolz.open(sav_dir + 'truth_im', mode='r')

import time
start = time.perf_counter()

#input_batch = X[:, :, :, :, 0]
#truth_batch = Y[:, :, :, :, 1]

input_batch = X[0:220]
truth_batch = Y[0:220, :, :, :, 1]
truth_batch = np.expand_dims(truth_batch, axis=-1)


input_batch = np.asarray(input_batch, np.float32)
truth_batch = np.asarray(truth_batch, np.float32)

stop = time.perf_counter()

diff = stop - start
print(diff)


# Split train and valid
# X_train = input_batch
# y_train = truth_batch
# X_valid = input_batch
# y_valid = truth_batch

X_train, X_valid, y_train, y_valid = train_test_split(input_batch, truth_batch, test_size=0.1, random_state=2018)




def conv2d_block(input_tensor, n_filters, kernel_size=3, batchnorm=True):
    # first layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same")(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    # second layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same")(x)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x



def get_unet_2d(input_img, n_filters=16, dropout=0.5, batchnorm=True):
    # contracting path
    c1 = conv2d_block(input_img, n_filters=n_filters*1, kernel_size=3, batchnorm=batchnorm)
    p1 = MaxPooling2D((2, 2)) (c1)
    p1 = Dropout(dropout*0.5)(p1)

    c2 = conv2d_block(p1, n_filters=n_filters*2, kernel_size=3, batchnorm=batchnorm)
    p2 = MaxPooling2D((2, 2)) (c2)
    p2 = Dropout(dropout)(p2)

    c3 = conv2d_block(p2, n_filters=n_filters*3, kernel_size=3, batchnorm=batchnorm)
    p3 = MaxPooling2D((2, 2)) (c3)
    p3 = Dropout(dropout)(p3)

    c4 = conv2d_block(p3, n_filters=n_filters*4, kernel_size=3, batchnorm=batchnorm)
    p4 = MaxPooling2D(pool_size=(2, 2)) (c4)
    p4 = Dropout(dropout)(p4)
    
    c5 = conv2d_block(p4, n_filters=n_filters*5, kernel_size=3, batchnorm=batchnorm)
    
    # expansive path
    u6 = Conv2DTranspose(n_filters*4, (3, 3), strides=(2, 2), padding='same') (c5)
    u6 = concatenate([u6, c4])
    u6 = Dropout(dropout)(u6)
    c6 = conv2d_block(u6, n_filters=n_filters*4, kernel_size=3, batchnorm=batchnorm)

    u7 = Conv2DTranspose(n_filters*4, (3, 3), strides=(2, 2), padding='same') (c6)
    u7 = concatenate([u7, c3])
    u7 = Dropout(dropout)(u7)
    c7 = conv2d_block(u7, n_filters=n_filters*3, kernel_size=3, batchnorm=batchnorm)

    u8 = Conv2DTranspose(n_filters*2, (3, 3), strides=(2, 2), padding='same') (c7)
    u8 = concatenate([u8, c2])
    u8 = Dropout(dropout)(u8)
    c8 = conv2d_block(u8, n_filters=n_filters*2, kernel_size=3, batchnorm=batchnorm)

    u9 = Conv2DTranspose(n_filters*1, (3, 3), strides=(2, 2), padding='same') (c8)
    u9 = concatenate([u9, c1], axis=3)
    u9 = Dropout(dropout)(u9)
    c9 = conv2d_block(u9, n_filters=n_filters*1, kernel_size=3, batchnorm=batchnorm)
    
    outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)
    model = Model(inputs=[input_img], outputs=[outputs])
    return model



def conv3d_block(input_tensor, n_filters, kernel_size=3, batchnorm=True):
    # first layer
    x = Conv3D(filters=n_filters, kernel_size=(kernel_size, kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same")(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    # second layer
    #x = Conv3D(filters=n_filters, kernel_size=(kernel_size, kernel_size, kernel_size), kernel_initializer="he_normal",
    #           padding="same")(x)
    if batchnorm:
        x = BatchNormalization()(x)
    #x = Activation("relu")(x)
    return x



def get_unet(input_img, n_filters=16, dropout=0.5, batchnorm=True):
    filt_siz = 5
    # contracting path
    c1 = conv3d_block(input_img, n_filters=n_filters*1, kernel_size=filt_siz, batchnorm=batchnorm)
    p1 = MaxPooling3D((2, 2, 2)) (c1)
    #p1 = Dropout(dropout*0.5)(p1)

    c2 = conv3d_block(p1, n_filters=n_filters*2, kernel_size=filt_siz, batchnorm=batchnorm)
    p2 = MaxPooling3D((2,2, 2)) (c2)
    #p2 = Dropout(dropout)(p2)

    c3 = conv3d_block(p2, n_filters=n_filters*3, kernel_size=filt_siz, batchnorm=batchnorm)
    p3 = MaxPooling3D((2,2, 2)) (c3)
    #p3 = Dropout(dropout)(p3)

    c4 = conv3d_block(p3, n_filters=n_filters*4, kernel_size=filt_siz, batchnorm=batchnorm)
    p4 = MaxPooling3D((2,2, 2)) (c4)
    #p4 = Dropout(dropout)(p4)
    
    c5 = conv3d_block(p4, n_filters=n_filters*5, kernel_size=filt_siz, batchnorm=batchnorm)
    
    # expansive path
    u6 = Conv3DTranspose(n_filters*4, (filt_siz, filt_siz, filt_siz), strides=(2,2, 2), padding='same') (c5)
    u6 = concatenate([u6, c4])
    #u6 = Dropout(dropout)(u6)
    c6 = conv3d_block(u6, n_filters=n_filters*4, kernel_size=filt_siz, batchnorm=batchnorm)

    u7 = Conv3DTranspose(n_filters*4, (filt_siz, filt_siz, filt_siz), strides=(2, 2, 2), padding='same') (c6)
    u7 = concatenate([u7, c3])
    #u7 = Dropout(dropout)(u7)
    c7 = conv3d_block(u7, n_filters=n_filters*3, kernel_size=filt_siz, batchnorm=batchnorm)

    u8 = Conv3DTranspose(n_filters*2, (filt_siz, filt_siz, filt_siz), strides=(2, 2, 2), padding='same') (c7)
    u8 = concatenate([u8, c2])
    #u8 = Dropout(dropout)(u8)
    c8 = conv3d_block(u8, n_filters=n_filters*2, kernel_size=filt_siz, batchnorm=batchnorm)

    u9 = Conv3DTranspose(n_filters*1, (filt_siz, filt_siz, filt_siz), strides=(2, 2, 2), padding='same') (c8)
    u9 = concatenate([u9, c1])
    #u9 = Dropout(dropout)(u9)
    c9 = conv3d_block(u9, n_filters=n_filters*1, kernel_size=filt_siz, batchnorm=batchnorm)
    
    outputs = Conv3D(1, (1, 1, 1), activation='sigmoid') (c9)
    model = Model(inputs=[input_img], outputs=[outputs])
    return model



input_img = Input((64, 256, 256, 1), name='img')
#input_img = Input((im_height, im_width, 1), name='img')
model = get_unet(input_img, n_filters=10, dropout=0, batchnorm=False)

model.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=["accuracy"])
model.summary()

callbacks = [
    EarlyStopping(patience=10, verbose=1),
    ReduceLROnPlateau(factor=0.1, patience=3, min_lr=0.00001, verbose=1),
    ModelCheckpoint('model-tgs-salt.h5', verbose=1, save_best_only=True, save_weights_only=True)
]


results = model.fit(X_train, y_train, batch_size=2, epochs=1, callbacks=callbacks,
                    validation_data=(X_valid, y_valid))



plt.figure(figsize=(8, 8))
plt.title("Learning curve")
plt.plot(results.history["loss"], label="loss")
plt.plot(results.history["val_loss"], label="val_loss")
plt.plot( np.argmin(results.history["val_loss"]), np.min(results.history["val_loss"]), marker="x", color="r", label="best model")
plt.xlabel("Epochs")
plt.ylabel("log_loss")
plt.legend();




# Load best model
model.load_weights('model-tgs-salt.h5')

# Evaluate on validation set (this must be equals to the best log_loss)
model.evaluate(X_valid, y_valid, verbose=1)





