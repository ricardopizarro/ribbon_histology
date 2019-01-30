import nibabel as nib
import numpy as np
import glob

import json

from keras import backend as K
from keras import regularizers
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization, UpSampling2D
from keras.layers import Conv2D, MaxPooling2D, merge, Input, Lambda, add, concatenate
from keras.optimizers import SGD,Adam
from scipy import stats
import itertools


def unet_101(input_shape, conv_size=(5,5),pool_size=(2,2), n_labels=1,
                  initial_learning_rate=0.00001, deconvolution=False, bias=False):
    """
    Builds the 2D UNet Keras model.
    :param input_shape: Shape of the input data (n_chanels, x_size, y_size, z_size). 
    :param downsize_filters_factor: Factor to which to reduce the number of filters. Making this value larger will
    reduce the amount of memory the model will need during training.
    :param pool_size: Pool size for the max pooling operations.
    :param n_labels: Number of binary labels that the model is learning.
    :param initial_learning_rate: Initial learning rate for the model. This will be decayed during training.
    :param deconvolution: If set to True, will use transpose convolution(deconvolution) instead of upsamping. This
    increases the amount memory required during training.
    :return: Untrained 2D UNet Model
    """
    inputs = Input(input_shape)
    # drop0 = Dropout(rate=0.5)(inputs)
    conv1 = Conv2D(10, conv_size, input_shape=input_shape, strides=pool_size, activation='relu',padding='same',use_bias=bias)(inputs)
    conv2 = Conv2D(20, conv_size, strides=pool_size, activation='relu', padding='same',use_bias=bias)(conv1)
    conv3 = Conv2D(30, conv_size, strides=pool_size, activation='relu', padding='same',use_bias=bias)(conv2)
    conv4 = Conv2D(40, conv_size, strides=pool_size, activation='relu', padding='same',use_bias=bias)(conv3)
    conv5 = Conv2D(50, conv_size, strides=pool_size, activation='relu', padding='same',use_bias=bias)(conv4)
    conv6 = Conv2D(60, conv_size, strides=pool_size, activation='relu', padding='same',use_bias=bias)(conv5)
    conv7 = Conv2D(70, conv_size, strides=pool_size, activation='relu', padding='same',use_bias=bias)(conv6)

    up8 = UpSampling2D(size=pool_size)(conv7)
    up8 = concatenate([up8, conv6], axis=3)
    conv8 = Conv2D(70, conv_size, activation='relu', padding='same',use_bias=bias)(up8)

    up9 = UpSampling2D(size=pool_size)(conv8)
    up9 = concatenate([up9, conv5], axis=3)
    conv9 = Conv2D(60, conv_size, activation='relu', padding='same',use_bias=bias)(up9)

    up10 = UpSampling2D(size=pool_size)(conv9)
    up10 = concatenate([up10, conv4], axis=3)
    conv10 = Conv2D(50, conv_size, activation='relu', padding='same',use_bias=bias)(up10)

    up11 = UpSampling2D(size=pool_size)(conv10)
    up11 = concatenate([up11, conv3], axis=3)
    conv11 = Conv2D(40, conv_size, activation='relu', padding='same',use_bias=bias)(up11)

    up12 = UpSampling2D(size=pool_size)(conv11)
    up12 = concatenate([up12, conv2], axis=3)
    conv12 = Conv2D(30, conv_size, activation='relu', padding='same',use_bias=bias)(up12)

    up13 = UpSampling2D(size=pool_size)(conv12)
    up13 = concatenate([up13, conv1], axis=3)
    conv13 = Conv2D(20, conv_size, activation='relu', padding='same',use_bias=bias)(up13)

    up14 = UpSampling2D(size=pool_size)(conv13)
    up14 = concatenate([up14, inputs], axis=3)
    conv14 = Conv2D(10, conv_size, activation='relu', padding='same',use_bias=bias)(up14)

    conv15 = Conv2D(n_labels, (1, 1),use_bias=bias)(conv14)
    act = Activation('softmax')(conv15)

    model = Model(inputs=inputs, outputs=act)
    print(model.summary())
    model.compile(optimizer=Adam(lr=1e-5), loss='categorical_crossentropy', metrics=[dice_coef])

    return model



def unet_101_bn(input_shape, conv_size=(5,5),pool_size=(2,2), n_labels=1,
                  initial_learning_rate=0.00001, deconvolution=False, bias=False):
    """
    Builds the 2D UNet Keras model.
    :param input_shape: Shape of the input data (n_chanels, x_size, y_size, z_size). 
    :param downsize_filters_factor: Factor to which to reduce the number of filters. Making this value larger will
    reduce the amount of memory the model will need during training.
    :param pool_size: Pool size for the max pooling operations.
    :param n_labels: Number of binary labels that the model is learning.
    :param initial_learning_rate: Initial learning rate for the model. This will be decayed during training.
    :param deconvolution: If set to True, will use transpose convolution(deconvolution) instead of upsamping. This
    increases the amount memory required during training.
    :return: Untrained 2D UNet Model
    """
    inputs = Input(input_shape)
    # drop0 = Dropout(rate=0.5)(inputs)
    conv1 = Conv2D(10, conv_size, input_shape=input_shape, strides=pool_size, activation='relu',padding='same',use_bias=bias)(inputs)
    bn1 = BatchNormalization()(conv1)
    conv2 = Conv2D(20, conv_size, strides=pool_size, activation='relu', padding='same',use_bias=bias)(bn1)
    bn2 = BatchNormalization()(conv2)
    conv3 = Conv2D(30, conv_size, strides=pool_size, activation='relu', padding='same',use_bias=bias)(bn2)
    bn3 = BatchNormalization()(conv3)
    conv4 = Conv2D(40, conv_size, strides=pool_size, activation='relu', padding='same',use_bias=bias)(bn3)
    bn4 = BatchNormalization()(conv4)
    conv5 = Conv2D(50, conv_size, strides=pool_size, activation='relu', padding='same',use_bias=bias)(bn4)
    bn5 = BatchNormalization()(conv5)
    conv6 = Conv2D(60, conv_size, strides=pool_size, activation='relu', padding='same',use_bias=bias)(bn5)
    bn6 = BatchNormalization()(conv6)
    conv7 = Conv2D(70, conv_size, strides=pool_size, activation='relu', padding='same',use_bias=bias)(bn6)
    bn7 = BatchNormalization()(conv7)

    up8 = UpSampling2D(size=pool_size)(bn7)
    up8 = concatenate([up8, bn6], axis=3)
    conv8 = Conv2D(70, conv_size, activation='relu', padding='same',use_bias=bias)(up8)
    bn8 = BatchNormalization()(conv8)

    up9 = UpSampling2D(size=pool_size)(bn8)
    up9 = concatenate([up9, bn5], axis=3)
    conv9 = Conv2D(60, conv_size, activation='relu', padding='same',use_bias=bias)(up9)
    bn9 = BatchNormalization()(conv9)

    up10 = UpSampling2D(size=pool_size)(bn9)
    up10 = concatenate([up10, bn4], axis=3)
    conv10 = Conv2D(50, conv_size, activation='relu', padding='same',use_bias=bias)(up10)
    bn10 = BatchNormalization()(conv10)

    up11 = UpSampling2D(size=pool_size)(bn10)
    up11 = concatenate([up11, bn3], axis=3)
    conv11 = Conv2D(40, conv_size, activation='relu', padding='same',use_bias=bias)(up11)
    bn11 = BatchNormalization()(conv11)

    up12 = UpSampling2D(size=pool_size)(bn11)
    up12 = concatenate([up12, bn2], axis=3)
    conv12 = Conv2D(30, conv_size, activation='relu', padding='same',use_bias=bias)(up12)
    bn12 = BatchNormalization()(conv12)

    up13 = UpSampling2D(size=pool_size)(bn12)
    up13 = concatenate([up13, bn1], axis=3)
    conv13 = Conv2D(20, conv_size, activation='relu', padding='same',use_bias=bias)(up13)
    bn13 = BatchNormalization()(conv13)

    up14 = UpSampling2D(size=pool_size)(bn13)
    up14 = concatenate([up14, inputs], axis=3)
    conv14 = Conv2D(10, conv_size, activation='relu', padding='same',use_bias=bias)(up14)
    bn14 = BatchNormalization()(conv14)

    conv15 = Conv2D(n_labels, (1, 1),use_bias=bias)(bn14)
    act = Activation('softmax')(conv15)

    model = Model(inputs=inputs, outputs=act)
    print(model.summary())
    model.compile(optimizer=Adam(lr=1e-5), loss='categorical_crossentropy', metrics=[dice_coef])

    return model


def unet_103(input_shape, conv_size=(5,5),pool_size=(4,4), n_labels=1,
                  initial_learning_rate=0.00001, deconvolution=False):
    """
    Builds the 2D UNet Keras model.
    :param input_shape: Shape of the input data (n_chanels, x_size, y_size, z_size). 
    :param downsize_filters_factor: Factor to which to reduce the number of filters. Making this value larger will
    reduce the amount of memory the model will need during training.
    :param pool_size: Pool size for the max pooling operations.
    :param n_labels: Number of binary labels that the model is learning.
    :param initial_learning_rate: Initial learning rate for the model. This will be decayed during training.
    :param deconvolution: If set to True, will use transpose convolution(deconvolution) instead of upsamping. This
    increases the amount memory required during training.
    :return: Untrained 2D UNet Model
    """
    inputs = Input(input_shape)
    # drop0 = Dropout(rate=0.5)(inputs)
    conv1 = Conv2D(10, conv_size, input_shape=input_shape, strides=pool_size, activation='relu',padding='same')(inputs)
    conv2 = Conv2D(20, conv_size, strides=pool_size, activation='relu', padding='same')(conv1)
    conv3 = Conv2D(30, conv_size, strides=pool_size, activation='relu', padding='same')(conv2)
    conv4 = Conv2D(40, conv_size, strides=pool_size, activation='relu', padding='same')(conv3)

    up5 = UpSampling2D(size=pool_size)(conv4)
    up5 = concatenate([up5, conv3], axis=3)
    conv5 = Conv2D(40, conv_size, activation='relu', padding='same')(up5)

    up6 = UpSampling2D(size=pool_size)(conv5)
    up6 = concatenate([up6, conv2], axis=3)
    conv6 = Conv2D(30, conv_size, activation='relu', padding='same')(up6)

    up7 = UpSampling2D(size=pool_size)(conv6)
    up7 = concatenate([up7, conv1], axis=3)
    conv7 = Conv2D(20, conv_size, activation='relu', padding='same')(up7)

    up8 = UpSampling2D(size=pool_size)(conv7)
    up8 = concatenate([up8, inputs], axis=3)
    conv8 = Conv2D(10, conv_size, activation='relu', padding='same')(up8)

    conv9 = Conv2D(n_labels, (1, 1))(conv8)
    act = Activation('softmax')(conv9)

    model = Model(inputs=inputs, outputs=act)
    print(model.summary())
    model.compile(optimizer=Adam(lr=1e-5), loss='categorical_crossentropy', metrics=[dice_coef])

    return model

def unet_103_drop(input_shape, conv_size=(5,5),pool_size=(4,4), n_labels=1,
                  initial_learning_rate=0.00001, deconvolution=False):
    """
    Builds the 2D UNet Keras model.
    :param input_shape: Shape of the input data (n_chanels, x_size, y_size, z_size). 
    :param downsize_filters_factor: Factor to which to reduce the number of filters. Making this value larger will
    reduce the amount of memory the model will need during training.
    :param pool_size: Pool size for the max pooling operations.
    :param n_labels: Number of binary labels that the model is learning.
    :param initial_learning_rate: Initial learning rate for the model. This will be decayed during training.
    :param deconvolution: If set to True, will use transpose convolution(deconvolution) instead of upsamping. This
    increases the amount memory required during training.
    :return: Untrained 2D UNet Model
    """
    inputs = Input(input_shape)
    # drop0 = Dropout(rate=0.5)(inputs)
    conv1 = Conv2D(10, conv_size, input_shape=input_shape, strides=pool_size, activation='relu',padding='same')(inputs)
    conv2 = Conv2D(20, conv_size, strides=pool_size, activation='relu', padding='same')(conv1)
    conv3 = Conv2D(30, conv_size, strides=pool_size, activation='relu', padding='same')(conv2)

    conv4a = Conv2D(40, conv_size, activation='relu', padding='same')(conv3)
    drop4a = Dropout(0.5)(conv4a)
    conv4b = Conv2D(40, conv_size, activation='relu', padding='same')(conv3)
    drop4b = Dropout(0.5)(conv4b)
    conv4c = Conv2D(40, conv_size, activation='relu', padding='same')(conv3)
    drop4c = Dropout(0.5)(conv4c)
    nadir = add([drop4a, drop4b, drop4c])

    # up5 = UpSampling2D(size=pool_size)(conv4)
    up5 = concatenate([nadir, conv3], axis=3)
    conv5 = Conv2D(40, conv_size, activation='relu', padding='same')(up5)

    up6 = UpSampling2D(size=pool_size)(conv5)
    up6 = concatenate([up6, conv2], axis=3)
    conv6 = Conv2D(30, conv_size, activation='relu', padding='same')(up6)

    up7 = UpSampling2D(size=pool_size)(conv6)
    up7 = concatenate([up7, conv1], axis=3)
    conv7 = Conv2D(20, conv_size, activation='relu', padding='same')(up7)

    up8 = UpSampling2D(size=pool_size)(conv7)
    up8 = concatenate([up8, inputs], axis=3)
    conv8 = Conv2D(10, conv_size, activation='relu', padding='same')(up8)

    conv9 = Conv2D(n_labels, (1, 1))(conv8)
    act = Activation('softmax')(conv9)

    model = Model(inputs=inputs, outputs=act)
    print(model.summary())
    model.compile(optimizer=Adam(lr=1e-5), loss='categorical_crossentropy', metrics=[dice_coef])

    return model


def unet_103_drop_bn(input_shape, conv_size=(5,5),pool_size=(4,4), n_labels=1,
                  initial_learning_rate=0.00001, deconvolution=False, bias=False):
    """
    Builds the 2D UNet Keras model.
    :param input_shape: Shape of the input data (n_chanels, x_size, y_size, z_size). 
    :param downsize_filters_factor: Factor to which to reduce the number of filters. Making this value larger will
    reduce the amount of memory the model will need during training.
    :param pool_size: Pool size for the max pooling operations.
    :param n_labels: Number of binary labels that the model is learning.
    :param initial_learning_rate: Initial learning rate for the model. This will be decayed during training.
    :param deconvolution: If set to True, will use transpose convolution(deconvolution) instead of upsamping. This
    increases the amount memory required during training.
    :return: Untrained 2D UNet Model
    """
    inputs = Input(input_shape)
    # drop0 = Dropout(rate=0.5)(inputs)
    conv1 = Conv2D(10, conv_size, input_shape=input_shape, strides=pool_size, activation='relu',padding='same', use_bias=bias)(inputs)
    bn1 = BatchNormalization()(conv1)
    conv2 = Conv2D(20, conv_size, strides=pool_size, activation='relu', padding='same', use_bias=bias)(bn1)
    bn2 = BatchNormalization()(conv2)
    conv3 = Conv2D(30, conv_size, strides=pool_size, activation='relu', padding='same', use_bias=bias)(bn2)
    bn3 = BatchNormalization()(conv3)

    conv4a = Conv2D(40, conv_size, activation='relu', padding='same', use_bias=bias)(bn3)
    bn4a = BatchNormalization()(conv4a)
    drop4a = Dropout(0.5)(bn4a)
    conv4b = Conv2D(40, conv_size, activation='relu', padding='same', use_bias=bias)(bn3)
    bn4b = BatchNormalization()(conv4b)
    drop4b = Dropout(0.5)(bn4b)
    conv4c = Conv2D(40, conv_size, activation='relu', padding='same', use_bias=bias)(bn3)
    bn4c = BatchNormalization()(conv4c)
    drop4c = Dropout(0.5)(bn4c)
    nadir = add([drop4a, drop4b, drop4c])

    # up5 = UpSampling2D(size=pool_size)(conv4)
    up5 = concatenate([nadir, bn3], axis=3)
    conv5 = Conv2D(40, conv_size, activation='relu', padding='same', use_bias=bias)(up5)
    bn5 = BatchNormalization()(conv5)

    up6 = UpSampling2D(size=pool_size)(bn5)
    up6 = concatenate([up6, bn2], axis=3)
    conv6 = Conv2D(30, conv_size, activation='relu', padding='same', use_bias=bias)(up6)
    bn6 = BatchNormalization()(conv6)

    up7 = UpSampling2D(size=pool_size)(bn6)
    up7 = concatenate([up7, bn1], axis=3)
    conv7 = Conv2D(20, conv_size, activation='relu', padding='same', use_bias=bias)(up7)
    bn7 = BatchNormalization()(conv7)

    up8 = UpSampling2D(size=pool_size)(bn7)
    up8 = concatenate([up8, inputs], axis=3)
    conv8 = Conv2D(10, conv_size, activation='relu', padding='same', use_bias=bias)(up8)
    bn8 = BatchNormalization()(conv8)

    conv9 = Conv2D(n_labels, (1, 1), use_bias=bias)(bn8)
    act = Activation('softmax')(conv9)

    model = Model(inputs=inputs, outputs=act)
    print(model.summary())
    model.compile(optimizer=Adam(lr=1e-5), loss='categorical_crossentropy', metrics=[dice_coef])

    return model

def unet_105(input_shape=(2430,2430,1), conv_size=(5,5),pool_size=(3,3), n_labels=1,
                  initial_learning_rate=0.00001, deconvolution=False):
    """
    Builds the 2D UNet Keras model.
    :param input_shape: Shape of the input data (n_chanels, x_size, y_size, z_size). 
    :param downsize_filters_factor: Factor to which to reduce the number of filters. Making this value larger will
    reduce the amount of memory the model will need during training.
    :param pool_size: Pool size for the max pooling operations.
    :param n_labels: Number of binary labels that the model is learning.
    :param initial_learning_rate: Initial learning rate for the model. This will be decayed during training.
    :param deconvolution: If set to True, will use transpose convolution(deconvolution) instead of upsamping. This
    increases the amount memory required during training.
    :return: Untrained 2D UNet Model
    """
    inputs = Input(input_shape)
    # drop0 = Dropout(rate=0.5)(inputs)
    conv1 = Conv2D(10, conv_size, input_shape=input_shape, strides=pool_size, activation='relu',padding='same')(inputs)
    conv2 = Conv2D(20, conv_size, strides=pool_size, activation='relu', padding='same')(conv1)
    conv3 = Conv2D(30, conv_size, strides=pool_size, activation='relu', padding='same')(conv2)
    conv4 = Conv2D(40, conv_size, strides=pool_size, activation='relu', padding='same')(conv3)
    conv5 = Conv2D(50, conv_size, strides=pool_size, activation='relu', padding='same')(conv4)

    up6 = UpSampling2D(size=pool_size)(conv5)
    up6 = concatenate([up6, conv4], axis=3)
    conv6 = Conv2D(50, conv_size, activation='relu', padding='same')(up6)

    up7 = UpSampling2D(size=pool_size)(conv6)
    up7 = concatenate([up7, conv3], axis=3)
    conv7 = Conv2D(40, conv_size, activation='relu', padding='same')(up7)

    up8 = UpSampling2D(size=pool_size)(conv7)
    up8 = concatenate([up8, conv2], axis=3)
    conv8 = Conv2D(30, conv_size, activation='relu', padding='same')(up8)

    up9 = UpSampling2D(size=pool_size)(conv8)
    up9 = concatenate([up9, conv1], axis=3)
    conv9 = Conv2D(20, conv_size, activation='relu', padding='same')(up9)

    up10 = UpSampling2D(size=pool_size)(conv9)
    up10 = concatenate([up10, inputs], axis=3)
    conv10 = Conv2D(10, conv_size, activation='relu', padding='same')(up10)

    conv11 = Conv2D(n_labels, (1, 1))(conv10)
    act = Activation('softmax')(conv11)

    model = Model(inputs=inputs, outputs=act)
    print(model.summary())
    model.compile(optimizer=Adam(lr=1e-5), loss='categorical_crossentropy', metrics=[dice_coef])

    return model



def unet_105_drop(input_shape=(2430,2430,1), conv_size=(5,5),pool_size=(3,3), n_labels=1,
                  initial_learning_rate=0.00001, deconvolution=False):
    """
    Builds the 2D UNet Keras model.
    :param input_shape: Shape of the input data (n_chanels, x_size, y_size, z_size). 
    :param downsize_filters_factor: Factor to which to reduce the number of filters. Making this value larger will
    reduce the amount of memory the model will need during training.
    :param pool_size: Pool size for the max pooling operations.
    :param n_labels: Number of binary labels that the model is learning.
    :param initial_learning_rate: Initial learning rate for the model. This will be decayed during training.
    :param deconvolution: If set to True, will use transpose convolution(deconvolution) instead of upsamping. This
    increases the amount memory required during training.
    :return: Untrained 2D UNet Model
    """
    inputs = Input(input_shape)
    # drop0 = Dropout(rate=0.5)(inputs)
    conv1 = Conv2D(10, conv_size, input_shape=input_shape, strides=pool_size, activation='relu',padding='same')(inputs)
    conv2 = Conv2D(20, conv_size, strides=pool_size, activation='relu', padding='same')(conv1)
    conv3 = Conv2D(30, conv_size, strides=pool_size, activation='relu', padding='same')(conv2)
    conv4 = Conv2D(40, conv_size, strides=pool_size, activation='relu', padding='same')(conv3)

    conv5a = Conv2D(50, conv_size, activation='relu', padding='same')(conv4)
    drop5a = Dropout(0.5)(conv5a)
    conv5b = Conv2D(50, conv_size, activation='relu', padding='same')(conv4)
    drop5b = Dropout(0.5)(conv5b)
    conv5c = Conv2D(50, conv_size, activation='relu', padding='same')(conv4)
    drop5c = Dropout(0.5)(conv5c)
    nadir = add([drop5a, drop5b, drop5c])

    # up5 = UpSampling2D(size=pool_size)(conv4)
    up6 = concatenate([nadir, conv4], axis=3)
    conv6 = Conv2D(50, conv_size, activation='relu', padding='same')(up6)

    up7 = UpSampling2D(size=pool_size)(conv6)
    up7 = concatenate([up7, conv3], axis=3)
    conv7 = Conv2D(40, conv_size, activation='relu', padding='same')(up7)

    up8 = UpSampling2D(size=pool_size)(conv7)
    up8 = concatenate([up8, conv2], axis=3)
    conv8 = Conv2D(30, conv_size, activation='relu', padding='same')(up8)

    up9 = UpSampling2D(size=pool_size)(conv8)
    up9 = concatenate([up9, conv1], axis=3)
    conv9 = Conv2D(20, conv_size, activation='relu', padding='same')(up9)

    up10 = UpSampling2D(size=pool_size)(conv9)
    up10 = concatenate([up10, inputs], axis=3)
    conv10 = Conv2D(10, conv_size, activation='relu', padding='same')(up10)

    conv11 = Conv2D(n_labels, (1, 1))(conv10)
    act = Activation('softmax')(conv11)

    model = Model(inputs=inputs, outputs=act)
    print(model.summary())
    model.compile(optimizer=Adam(lr=1e-5), loss='categorical_crossentropy', metrics=[dice_coef])

    return model

def unet_107(input_shape=(2500,2500,1), conv_size=(5,5),pool_size=(5,5), n_labels=1,
                  initial_learning_rate=0.00001, deconvolution=False):
    """
    Builds the 2D UNet Keras model.
    :param input_shape: Shape of the input data (n_chanels, x_size, y_size, z_size). 
    :param downsize_filters_factor: Factor to which to reduce the number of filters. Making this value larger will
    reduce the amount of memory the model will need during training.
    :param pool_size: Pool size for the max pooling operations.
    :param n_labels: Number of binary labels that the model is learning.
    :param initial_learning_rate: Initial learning rate for the model. This will be decayed during training.
    :param deconvolution: If set to True, will use transpose convolution(deconvolution) instead of upsamping. This
    increases the amount memory required during training.
    :return: Untrained 2D UNet Model
    """
    inputs = Input(input_shape)
    # drop0 = Dropout(rate=0.5)(inputs)
    conv1 = Conv2D(10, conv_size, input_shape=input_shape, strides=pool_size, activation='relu',padding='same')(inputs)
    conv2 = Conv2D(20, conv_size, strides=pool_size, activation='relu', padding='same')(conv1)
    conv3 = Conv2D(30, conv_size, strides=pool_size, activation='relu', padding='same')(conv2)

    up4 = UpSampling2D(size=pool_size)(conv3)
    up4 = concatenate([up4, conv2], axis=3)
    conv4 = Conv2D(50, conv_size, activation='relu', padding='same')(up4)

    up5 = UpSampling2D(size=pool_size)(conv4)
    up5 = concatenate([up5, conv1], axis=3)
    conv5 = Conv2D(40, conv_size, activation='relu', padding='same')(up5)

    up6 = UpSampling2D(size=pool_size)(conv5)
    up6 = concatenate([up6, inputs], axis=3)
    conv6 = Conv2D(30, conv_size, activation='relu', padding='same')(up6)

    conv7 = Conv2D(n_labels, (1, 1))(conv6)
    act = Activation('softmax')(conv7)

    model = Model(inputs=inputs, outputs=act)
    print(model.summary())
    model.compile(optimizer=Adam(lr=1e-5), loss='categorical_crossentropy', metrics=[dice_coef])

    return model


def unet_107_bn(input_shape=(2500,2500,1), conv_size=(5,5),pool_size=(5,5), n_labels=1,
                  initial_learning_rate=0.00001, deconvolution=False,bias=False):
    """
    Builds the 2D UNet Keras model.
    :param input_shape: Shape of the input data (n_chanels, x_size, y_size, z_size). 
    :param downsize_filters_factor: Factor to which to reduce the number of filters. Making this value larger will
    reduce the amount of memory the model will need during training.
    :param pool_size: Pool size for the max pooling operations.
    :param n_labels: Number of binary labels that the model is learning.
    :param initial_learning_rate: Initial learning rate for the model. This will be decayed during training.
    :param deconvolution: If set to True, will use transpose convolution(deconvolution) instead of upsamping. This
    increases the amount memory required during training.
    :return: Untrained 2D UNet Model
    """
    inputs = Input(input_shape)
    # drop0 = Dropout(rate=0.5)(inputs)
    conv1 = Conv2D(10, conv_size, input_shape=input_shape, strides=pool_size, activation='relu',padding='same', use_bias=bias)(inputs)
    bn1 = BatchNormalization()(conv1)
    conv2 = Conv2D(20, conv_size, strides=pool_size, activation='relu', padding='same', use_bias=bias)(bn1)
    bn2 = BatchNormalization()(conv2)
    conv3 = Conv2D(30, conv_size, strides=pool_size, activation='relu', padding='same', use_bias=bias)(bn2)
    bn3 = BatchNormalization()(conv3)

    up4 = UpSampling2D(size=pool_size)(bn3)
    up4 = concatenate([up4, bn2], axis=3)
    conv4 = Conv2D(50, conv_size, activation='relu', padding='same', use_bias=bias)(up4)
    bn4 = BatchNormalization()(conv4)

    up5 = UpSampling2D(size=pool_size)(bn4)
    up5 = concatenate([up5, bn1], axis=3)
    conv5 = Conv2D(40, conv_size, activation='relu', padding='same', use_bias=bias)(up5)
    bn5 = BatchNormalization()(conv5)

    up6 = UpSampling2D(size=pool_size)(bn5)
    up6 = concatenate([up6, inputs], axis=3)
    conv6 = Conv2D(30, conv_size, activation='relu', padding='same', use_bias=bias)(up6)
    bn6 = BatchNormalization()(conv6)

    conv7 = Conv2D(n_labels, (1, 1))(bn6)
    act = Activation('softmax')(conv7)

    model = Model(inputs=inputs, outputs=act)
    print(model.summary())
    model.compile(optimizer=Adam(lr=1e-5), loss='categorical_crossentropy', metrics=[dice_coef])

    return model


def unet_107_drop(input_shape=(2500,2500,1), conv_size=(5,5),pool_size=(5,5), n_labels=1,
                  initial_learning_rate=0.00001, deconvolution=False):
    """
    Builds the 2D UNet Keras model.
    :param input_shape: Shape of the input data (n_chanels, x_size, y_size, z_size). 
    :param downsize_filters_factor: Factor to which to reduce the number of filters. Making this value larger will
    reduce the amount of memory the model will need during training.
    :param pool_size: Pool size for the max pooling operations.
    :param n_labels: Number of binary labels that the model is learning.
    :param initial_learning_rate: Initial learning rate for the model. This will be decayed during training.
    :param deconvolution: If set to True, will use transpose convolution(deconvolution) instead of upsamping. This
    increases the amount memory required during training.
    :return: Untrained 2D UNet Model
    """
    inputs = Input(input_shape)
    # drop0 = Dropout(rate=0.5)(inputs)
    conv1 = Conv2D(10, conv_size, input_shape=input_shape, strides=pool_size, activation='relu',padding='same')(inputs)
    conv2 = Conv2D(20, conv_size, strides=pool_size, activation='relu', padding='same')(conv1)
    # conv3 = Conv2D(30, conv_size, strides=pool_size, activation='relu', padding='same')(conv2)

    conv3a = Conv2D(30, conv_size, activation='relu', padding='same')(conv2)
    drop3a = Dropout(0.5)(conv3a)
    conv3b = Conv2D(30, conv_size, activation='relu', padding='same')(conv2)
    drop3b = Dropout(0.5)(conv3b)
    conv3c = Conv2D(30, conv_size, activation='relu', padding='same')(conv2)
    drop3c = Dropout(0.5)(conv3c)
    nadir = add([drop3a, drop3b, drop3c])

    # up4 = UpSampling2D(size=pool_size)(conv3)
    up4 = concatenate([nadir, conv2], axis=3)
    conv4 = Conv2D(50, conv_size, activation='relu', padding='same')(up4)

    up5 = UpSampling2D(size=pool_size)(conv4)
    up5 = concatenate([up5, conv1], axis=3)
    conv5 = Conv2D(40, conv_size, activation='relu', padding='same')(up5)

    up6 = UpSampling2D(size=pool_size)(conv5)
    up6 = concatenate([up6, inputs], axis=3)
    conv6 = Conv2D(30, conv_size, activation='relu', padding='same')(up6)

    conv7 = Conv2D(n_labels, (1, 1))(conv6)
    act = Activation('softmax')(conv7)

    model = Model(inputs=inputs, outputs=act)
    print(model.summary())
    model.compile(optimizer=Adam(lr=1e-5), loss='categorical_crossentropy', metrics=[dice_coef])

    return model




def dice_coef(y_true, y_pred, smooth=1.):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


input_shape=(2560,2560,1)
model = unet_101(input_shape=input_shape, conv_size=(5,5),pool_size=(2,2), n_labels=2, bias=False)
json_string = model.to_json()
fn = "../model/model.unet.v101.json"
print('Writing to file, model : {}'.format(fn))
with open(fn, 'w') as outfile:
    json.dump(json_string, outfile)


input_shape=(2560,2560,1)
model = unet_101(input_shape=input_shape, conv_size=(5,5),pool_size=(2,2), n_labels=2, bias=True)
json_string = model.to_json()
fn = "../model/model.unet.v101_bias.json"
print('Writing to file, model : {}'.format(fn))
with open(fn, 'w') as outfile:
    json.dump(json_string, outfile)


input_shape=(2560,2560,1)
model = unet_101_bn(input_shape=input_shape, conv_size=(5,5),pool_size=(2,2), n_labels=2, bias=False)
json_string = model.to_json()
fn = "../model/model.unet.v101_bn.json"
print('Writing to file, model : {}'.format(fn))
with open(fn, 'w') as outfile:
    json.dump(json_string, outfile)


input_shape=(2560,2560,1)
model = unet_101_bn(input_shape=input_shape, conv_size=(5,5),pool_size=(2,2), n_labels=2, bias=True)
json_string = model.to_json()
fn = "../model/model.unet.v101_bn_bias.json"
print('Writing to file, model : {}'.format(fn))
with open(fn, 'w') as outfile:
    json.dump(json_string, outfile)


input_shape=(2560,2560,1)
model = unet_103(input_shape=input_shape, conv_size=(5,5),pool_size=(4,4), n_labels=2)
json_string = model.to_json()
fn = "../model/model.unet.v103.json"
print('Writing to file, model : {}'.format(fn))
with open(fn, 'w') as outfile:
    json.dump(json_string, outfile)


input_shape=(2560,2560,1)
model = unet_103_drop(input_shape=input_shape, conv_size=(5,5),pool_size=(4,4), n_labels=2)
json_string = model.to_json()
fn = "../model/model.unet.v103_drop.json"
print('Writing to file, model : {}'.format(fn))
with open(fn, 'w') as outfile:
    json.dump(json_string, outfile)


input_shape=(2560,2560,1)
model = unet_103_drop_bn(input_shape=input_shape, conv_size=(5,5),pool_size=(4,4), n_labels=2,bias=True)
json_string = model.to_json()
fn = "../model/model.unet.v103_drop_bn_bias.json"
print('Writing to file, model : {}'.format(fn))
with open(fn, 'w') as outfile:
    json.dump(json_string, outfile)


input_shape=(2430,2430,1)
model = unet_105(input_shape=input_shape, conv_size=(5,5),pool_size=(3,3), n_labels=2)
json_string = model.to_json()
fn = "../model/model.unet.v105.json"
print('Writing to file, model : {}'.format(fn))
with open(fn, 'w') as outfile:
    json.dump(json_string, outfile)


input_shape=(2430,2430,1)
model = unet_105_drop(input_shape=input_shape, conv_size=(5,5),pool_size=(3,3), n_labels=2)
json_string = model.to_json()
fn = "../model/model.unet.v105_drop.json"
print('Writing to file, model : {}'.format(fn))
with open(fn, 'w') as outfile:
    json.dump(json_string, outfile)


input_shape=(2500,2500,1)
model = unet_107(input_shape=input_shape, conv_size=(5,5),pool_size=(5,5), n_labels=2)
json_string = model.to_json()
fn = "../model/model.unet.v107.json"
print('Writing to file, model : {}'.format(fn))
with open(fn, 'w') as outfile:
    json.dump(json_string, outfile)


input_shape=(2500,2500,1)
model = unet_107_bn(input_shape=input_shape, conv_size=(5,5),pool_size=(5,5), n_labels=2,bias=True)
json_string = model.to_json()
fn = "../model/model.unet.v107_bn_bias.json"
print('Writing to file, model : {}'.format(fn))
with open(fn, 'w') as outfile:
    json.dump(json_string, outfile)


input_shape=(2500,2500,1)
model = unet_107_drop(input_shape=input_shape, conv_size=(5,5),pool_size=(5,5), n_labels=2)
json_string = model.to_json()
fn = "../model/model.unet.v107_drop.json"
print('Writing to file, model : {}'.format(fn))
with open(fn, 'w') as outfile:
    json.dump(json_string, outfile)


