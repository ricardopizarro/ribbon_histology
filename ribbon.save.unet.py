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



def unet_101_pad(input_shape, conv_size=(5,5),pool_size=(2,2), n_labels=1,
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
    conv1_001 = Conv2D(10, conv_size, input_shape=input_shape, activation='relu',padding='same',use_bias=bias)(inputs)
    conv1 = Conv2D(10, conv_size, input_shape=input_shape, strides=pool_size, activation='relu',padding='same',use_bias=bias)(conv1_001)
    conv2_001 = Conv2D(20, conv_size, activation='relu', padding='same',use_bias=bias)(conv1)
    conv2 = Conv2D(20, conv_size, strides=pool_size, activation='relu', padding='same',use_bias=bias)(conv2_001)
    conv3_001 = Conv2D(30, conv_size, activation='relu', padding='same',use_bias=bias)(conv2)
    conv3 = Conv2D(30, conv_size, strides=pool_size, activation='relu', padding='same',use_bias=bias)(conv3_001)
    conv4_001 = Conv2D(40, conv_size, activation='relu', padding='same',use_bias=bias)(conv3)
    conv4 = Conv2D(40, conv_size, strides=pool_size, activation='relu', padding='same',use_bias=bias)(conv4_001)
    conv5_001 = Conv2D(50, conv_size, activation='relu', padding='same',use_bias=bias)(conv4)
    conv5 = Conv2D(50, conv_size, strides=pool_size, activation='relu', padding='same',use_bias=bias)(conv5_001)
    conv6_001 = Conv2D(60, conv_size, activation='relu', padding='same',use_bias=bias)(conv5)
    conv6 = Conv2D(60, conv_size, strides=pool_size, activation='relu', padding='same',use_bias=bias)(conv6_001)
    conv7_001 = Conv2D(70, conv_size, activation='relu', padding='same',use_bias=bias)(conv6)
    conv7 = Conv2D(70, conv_size, strides=pool_size, activation='relu', padding='same',use_bias=bias)(conv7_001)

    up8 = UpSampling2D(size=pool_size)(conv7)
    up8 = concatenate([up8, conv6], axis=3)
    conv8_001 = Conv2D(70, conv_size, activation='relu', padding='same',use_bias=bias)(up8)
    conv8 = Conv2D(70, conv_size, activation='relu', padding='same',use_bias=bias)(conv8_001)

    up9 = UpSampling2D(size=pool_size)(conv8)
    up9 = concatenate([up9, conv5], axis=3)
    conv9_001 = Conv2D(60, conv_size, activation='relu', padding='same',use_bias=bias)(up9)
    conv9 = Conv2D(60, conv_size, activation='relu', padding='same',use_bias=bias)(conv9_001)

    up10 = UpSampling2D(size=pool_size)(conv9)
    up10 = concatenate([up10, conv4], axis=3)
    conv10_001 = Conv2D(50, conv_size, activation='relu', padding='same',use_bias=bias)(up10)
    conv10 = Conv2D(50, conv_size, activation='relu', padding='same',use_bias=bias)(conv10_001)

    up11 = UpSampling2D(size=pool_size)(conv10)
    up11 = concatenate([up11, conv3], axis=3)
    conv11_001 = Conv2D(40, conv_size, activation='relu', padding='same',use_bias=bias)(up11)
    conv11 = Conv2D(40, conv_size, activation='relu', padding='same',use_bias=bias)(conv11_001)

    up12 = UpSampling2D(size=pool_size)(conv11)
    up12 = concatenate([up12, conv2], axis=3)
    conv12_001 = Conv2D(30, conv_size, activation='relu', padding='same',use_bias=bias)(up12)
    conv12 = Conv2D(30, conv_size, activation='relu', padding='same',use_bias=bias)(conv12_001)

    up13 = UpSampling2D(size=pool_size)(conv12)
    up13 = concatenate([up13, conv1], axis=3)
    conv13_001 = Conv2D(20, conv_size, activation='relu', padding='same',use_bias=bias)(up13)
    conv13 = Conv2D(20, conv_size, activation='relu', padding='same',use_bias=bias)(conv13_001)

    up14 = UpSampling2D(size=pool_size)(conv13)
    up14 = concatenate([up14, inputs], axis=3)
    conv14_001 = Conv2D(10, conv_size, activation='relu', padding='same',use_bias=bias)(up14)
    conv14 = Conv2D(10, conv_size, activation='relu', padding='same',use_bias=bias)(conv14_001)

    conv15 = Conv2D(n_labels, (1, 1),use_bias=bias)(conv14)
    act = Activation('softmax')(conv15)

    model = Model(inputs=inputs, outputs=act)
    print(model.summary())
    model.compile(optimizer=Adam(lr=1e-5), loss='categorical_crossentropy', metrics=[dice_coef])

    return model



def unet_101_pad2(input_shape, conv_size=(5,5),pool_size=(2,2), n_labels=1,
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
    conv1_001 = Conv2D(10, conv_size, input_shape=input_shape, activation='relu',padding='same',use_bias=bias)(inputs)
    conv1_002 = Conv2D(10, conv_size, input_shape=input_shape, activation='relu',padding='same',use_bias=bias)(conv1_001)
    conv1 = Conv2D(10, conv_size, input_shape=input_shape, strides=pool_size, activation='relu',padding='same',use_bias=bias)(conv1_002)
    conv2_001 = Conv2D(20, conv_size, activation='relu', padding='same',use_bias=bias)(conv1)
    conv2_002 = Conv2D(20, conv_size, activation='relu', padding='same',use_bias=bias)(conv2_001)
    conv2 = Conv2D(20, conv_size, strides=pool_size, activation='relu', padding='same',use_bias=bias)(conv2_002)
    conv3_001 = Conv2D(30, conv_size, activation='relu', padding='same',use_bias=bias)(conv2)
    conv3_002 = Conv2D(30, conv_size, activation='relu', padding='same',use_bias=bias)(conv3_001)
    conv3 = Conv2D(30, conv_size, strides=pool_size, activation='relu', padding='same',use_bias=bias)(conv3_002)
    conv4_001 = Conv2D(40, conv_size, activation='relu', padding='same',use_bias=bias)(conv3)
    conv4_002 = Conv2D(40, conv_size, activation='relu', padding='same',use_bias=bias)(conv4_001)
    conv4 = Conv2D(40, conv_size, strides=pool_size, activation='relu', padding='same',use_bias=bias)(conv4_002)
    conv5_001 = Conv2D(50, conv_size, activation='relu', padding='same',use_bias=bias)(conv4)
    conv5_002 = Conv2D(50, conv_size, activation='relu', padding='same',use_bias=bias)(conv5_001)
    conv5 = Conv2D(50, conv_size, strides=pool_size, activation='relu', padding='same',use_bias=bias)(conv5_002)
    conv6_001 = Conv2D(60, conv_size, activation='relu', padding='same',use_bias=bias)(conv5)
    conv6_002 = Conv2D(60, conv_size, activation='relu', padding='same',use_bias=bias)(conv6_001)
    conv6 = Conv2D(60, conv_size, strides=pool_size, activation='relu', padding='same',use_bias=bias)(conv6_002)
    conv7_001 = Conv2D(70, conv_size, activation='relu', padding='same',use_bias=bias)(conv6)
    conv7_002 = Conv2D(70, conv_size, activation='relu', padding='same',use_bias=bias)(conv7_001)
    conv7 = Conv2D(70, conv_size, strides=pool_size, activation='relu', padding='same',use_bias=bias)(conv7_002)

    up8 = UpSampling2D(size=pool_size)(conv7)
    up8 = concatenate([up8, conv6], axis=3)
    conv8_001 = Conv2D(70, conv_size, activation='relu', padding='same',use_bias=bias)(up8)
    conv8_002 = Conv2D(70, conv_size, activation='relu', padding='same',use_bias=bias)(conv8_001)
    conv8 = Conv2D(70, conv_size, activation='relu', padding='same',use_bias=bias)(conv8_002)

    up9 = UpSampling2D(size=pool_size)(conv8)
    up9 = concatenate([up9, conv5], axis=3)
    conv9_001 = Conv2D(60, conv_size, activation='relu', padding='same',use_bias=bias)(up9)
    conv9_002 = Conv2D(60, conv_size, activation='relu', padding='same',use_bias=bias)(conv9_001)
    conv9 = Conv2D(60, conv_size, activation='relu', padding='same',use_bias=bias)(conv9_002)

    up10 = UpSampling2D(size=pool_size)(conv9)
    up10 = concatenate([up10, conv4], axis=3)
    conv10_001 = Conv2D(50, conv_size, activation='relu', padding='same',use_bias=bias)(up10)
    conv10_002 = Conv2D(50, conv_size, activation='relu', padding='same',use_bias=bias)(conv10_001)
    conv10 = Conv2D(50, conv_size, activation='relu', padding='same',use_bias=bias)(conv10_002)

    up11 = UpSampling2D(size=pool_size)(conv10)
    up11 = concatenate([up11, conv3], axis=3)
    conv11_001 = Conv2D(40, conv_size, activation='relu', padding='same',use_bias=bias)(up11)
    conv11_002 = Conv2D(40, conv_size, activation='relu', padding='same',use_bias=bias)(conv11_001)
    conv11 = Conv2D(40, conv_size, activation='relu', padding='same',use_bias=bias)(conv11_002)

    up12 = UpSampling2D(size=pool_size)(conv11)
    up12 = concatenate([up12, conv2], axis=3)
    conv12_001 = Conv2D(30, conv_size, activation='relu', padding='same',use_bias=bias)(up12)
    conv12_002 = Conv2D(30, conv_size, activation='relu', padding='same',use_bias=bias)(conv12_001)
    conv12 = Conv2D(30, conv_size, activation='relu', padding='same',use_bias=bias)(conv12_002)

    up13 = UpSampling2D(size=pool_size)(conv12)
    up13 = concatenate([up13, conv1], axis=3)
    conv13_001 = Conv2D(20, conv_size, activation='relu', padding='same',use_bias=bias)(up13)
    conv13_002 = Conv2D(20, conv_size, activation='relu', padding='same',use_bias=bias)(conv13_001)
    conv13 = Conv2D(20, conv_size, activation='relu', padding='same',use_bias=bias)(conv13_002)

    up14 = UpSampling2D(size=pool_size)(conv13)
    up14 = concatenate([up14, inputs], axis=3)
    conv14_001 = Conv2D(10, conv_size, activation='relu', padding='same',use_bias=bias)(up14)
    conv14_002 = Conv2D(10, conv_size, activation='relu', padding='same',use_bias=bias)(conv14_001)
    conv14 = Conv2D(10, conv_size, activation='relu', padding='same',use_bias=bias)(conv14_002)

    conv15 = Conv2D(n_labels, (1, 1),use_bias=bias)(conv14)
    act = Activation('softmax')(conv15)

    model = Model(inputs=inputs, outputs=act)
    print(model.summary())
    model.compile(optimizer=Adam(lr=1e-5), loss='categorical_crossentropy', metrics=[dice_coef])

    return model



def unet_101_drop(input_shape, conv_size=(5,5),pool_size=(2,2), n_labels=1,
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

    # conv7 = Conv2D(70, conv_size, strides=pool_size, activation='relu', padding='same',use_bias=bias)(conv6)
    conv7a = Conv2D(70, conv_size, activation='relu', padding='same')(conv6)
    drop7a = Dropout(0.5)(conv7a)
    conv7b = Conv2D(70, conv_size, activation='relu', padding='same')(conv6)
    drop7b = Dropout(0.5)(conv7b)
    conv7c = Conv2D(70, conv_size, activation='relu', padding='same')(conv6)
    drop7c = Dropout(0.5)(conv7c)
    nadir = add([drop7a, drop7b, drop7c])

    # up8 = UpSampling2D(size=pool_size)(conv7)
    up8 = concatenate([nadir, conv6], axis=3)
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



def unet_101_drop2(input_shape, conv_size=(5,5),pool_size=(2,2), n_labels=1,
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

    # conv7 = Conv2D(70, conv_size, strides=pool_size, activation='relu', padding='same',use_bias=bias)(conv6)
    conv7a = Conv2D(70, conv_size, strides=pool_size, activation='relu', padding='same')(conv6)
    drop7a = Dropout(0.5)(conv7a)
    conv7b = Conv2D(70, conv_size, strides=pool_size, activation='relu', padding='same')(conv6)
    drop7b = Dropout(0.5)(conv7b)
    conv7c = Conv2D(70, conv_size, strides=pool_size, activation='relu', padding='same')(conv6)
    drop7c = Dropout(0.5)(conv7c)
    nadir = add([drop7a, drop7b, drop7c])

    up8 = UpSampling2D(size=pool_size)(nadir)
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

def unet_103_drop2(input_shape, conv_size=(5,5),pool_size=(4,4), n_labels=1,
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

    conv4a = Conv2D(40, conv_size, strides=pool_size, activation='relu', padding='same')(conv3)
    drop4a = Dropout(0.5)(conv4a)
    conv4b = Conv2D(40, conv_size, strides=pool_size, activation='relu', padding='same')(conv3)
    drop4b = Dropout(0.5)(conv4b)
    conv4c = Conv2D(40, conv_size, strides=pool_size, activation='relu', padding='same')(conv3)
    drop4c = Dropout(0.5)(conv4c)
    nadir = add([drop4a, drop4b, drop4c])

    up5 = UpSampling2D(size=pool_size)(nadir)
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


def unet_109(input_shape, conv_size=(5,5),pool_size=(8,8), n_labels=1,
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
    conv4 = Conv2D(30, conv_size, activation='relu', padding='same')(up4)

    up5 = UpSampling2D(size=pool_size)(conv4)
    up5 = concatenate([up5, conv1], axis=3)
    conv5 = Conv2D(20, conv_size, activation='relu', padding='same')(up5)

    up6 = UpSampling2D(size=pool_size)(conv5)
    up6 = concatenate([up6, inputs], axis=3)
    conv6 = Conv2D(10, conv_size, activation='relu', padding='same')(up6)

    conv8 = Conv2D(n_labels, (1, 1))(conv6)
    act = Activation('softmax')(conv8)

    model = Model(inputs=inputs, outputs=act)
    print(model.summary())
    model.compile(optimizer=Adam(lr=1e-5), loss='categorical_crossentropy', metrics=[dice_coef])

    return model



def unet_109_drop(input_shape, conv_size=(5,5),pool_size=(8,8), n_labels=1,
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

    conv3a = Conv2D(40, conv_size, activation='relu', padding='same')(conv2)
    drop3a = Dropout(0.5)(conv3a)
    conv3b = Conv2D(40, conv_size, activation='relu', padding='same')(conv2)
    drop3b = Dropout(0.5)(conv3b)
    conv3c = Conv2D(40, conv_size, activation='relu', padding='same')(conv2)
    drop3c = Dropout(0.5)(conv3c)
    nadir = add([drop3a, drop3b, drop3c])

    # up4 = UpSampling2D(size=pool_size)(conv3)
    up4 = concatenate([nadir, conv2], axis=3)
    conv4 = Conv2D(30, conv_size, activation='relu', padding='same')(up4)

    up5 = UpSampling2D(size=pool_size)(conv4)
    up5 = concatenate([up5, conv1], axis=3)
    conv5 = Conv2D(20, conv_size, activation='relu', padding='same')(up5)

    up6 = UpSampling2D(size=pool_size)(conv5)
    up6 = concatenate([up6, inputs], axis=3)
    conv6 = Conv2D(10, conv_size, activation='relu', padding='same')(up6)

    conv8 = Conv2D(n_labels, (1, 1))(conv6)
    act = Activation('softmax')(conv8)

    model = Model(inputs=inputs, outputs=act)
    print(model.summary())
    model.compile(optimizer=Adam(lr=1e-5), loss='categorical_crossentropy', metrics=[dice_coef])

    return model


def unet_109_drop2(input_shape, conv_size=(5,5),pool_size=(8,8), n_labels=1,
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

    conv3a = Conv2D(40, conv_size, strides=pool_size, activation='relu', padding='same')(conv2)
    drop3a = Dropout(0.5)(conv3a)
    conv3b = Conv2D(40, conv_size, strides=pool_size, activation='relu', padding='same')(conv2)
    drop3b = Dropout(0.5)(conv3b)
    conv3c = Conv2D(40, conv_size, strides=pool_size, activation='relu', padding='same')(conv2)
    drop3c = Dropout(0.5)(conv3c)
    nadir = add([drop3a, drop3b, drop3c])

    up4 = UpSampling2D(size=pool_size)(nadir)
    up4 = concatenate([up4, conv2], axis=3)
    conv4 = Conv2D(30, conv_size, activation='relu', padding='same')(up4)

    up5 = UpSampling2D(size=pool_size)(conv4)
    up5 = concatenate([up5, conv1], axis=3)
    conv5 = Conv2D(20, conv_size, activation='relu', padding='same')(up5)

    up6 = UpSampling2D(size=pool_size)(conv5)
    up6 = concatenate([up6, inputs], axis=3)
    conv6 = Conv2D(10, conv_size, activation='relu', padding='same')(up6)

    conv8 = Conv2D(n_labels, (1, 1))(conv6)
    act = Activation('softmax')(conv8)

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


model = unet_101_pad(input_shape=input_shape, conv_size=(5,5),pool_size=(2,2), n_labels=2, bias=False)
json_string = model.to_json()
fn = "../model/model.unet.v101_pad.json"
print('Writing to file, model : {}'.format(fn))
with open(fn, 'w') as outfile:
    json.dump(json_string, outfile)


model = unet_101_pad2(input_shape=input_shape, conv_size=(5,5),pool_size=(2,2), n_labels=2, bias=False)
json_string = model.to_json()
fn = "../model/model.unet.v101_pad2.json"
print('Writing to file, model : {}'.format(fn))
with open(fn, 'w') as outfile:
    json.dump(json_string, outfile)


model = unet_101_drop(input_shape=input_shape, conv_size=(5,5),pool_size=(2,2), n_labels=2, bias=False)
json_string = model.to_json()
fn = "../model/model.unet.v101_drop.json"
print('Writing to file, model : {}'.format(fn))
with open(fn, 'w') as outfile:
    json.dump(json_string, outfile)


model = unet_101_drop2(input_shape=input_shape, conv_size=(5,5),pool_size=(2,2), n_labels=2, bias=False)
json_string = model.to_json()
fn = "../model/model.unet.v101_drop2.json"
print('Writing to file, model : {}'.format(fn))
with open(fn, 'w') as outfile:
    json.dump(json_string, outfile)


model = unet_103(input_shape=input_shape, conv_size=(5,5),pool_size=(4,4), n_labels=2)
json_string = model.to_json()
fn = "../model/model.unet.v103.json"
print('Writing to file, model : {}'.format(fn))
with open(fn, 'w') as outfile:
    json.dump(json_string, outfile)


model = unet_103_drop(input_shape=input_shape, conv_size=(5,5),pool_size=(4,4), n_labels=2)
json_string = model.to_json()
fn = "../model/model.unet.v103_drop.json"
print('Writing to file, model : {}'.format(fn))
with open(fn, 'w') as outfile:
    json.dump(json_string, outfile)


model = unet_103_drop2(input_shape=input_shape, conv_size=(5,5),pool_size=(4,4), n_labels=2)
json_string = model.to_json()
fn = "../model/model.unet.v103_drop2.json"
print('Writing to file, model : {}'.format(fn))
with open(fn, 'w') as outfile:
    json.dump(json_string, outfile)


model = unet_109(input_shape=input_shape, conv_size=(5,5),pool_size=(8,8), n_labels=2)
json_string = model.to_json()
fn = "../model/model.unet.v109.json"
print('Writing to file, model : {}'.format(fn))
with open(fn, 'w') as outfile:
    json.dump(json_string, outfile)


model = unet_109_drop(input_shape=input_shape, conv_size=(5,5),pool_size=(8,8), n_labels=2)
json_string = model.to_json()
fn = "../model/model.unet.v109_drop.json"
print('Writing to file, model : {}'.format(fn))
with open(fn, 'w') as outfile:
    json.dump(json_string, outfile)


model = unet_109_drop2(input_shape=input_shape, conv_size=(5,5),pool_size=(8,8), n_labels=2)
json_string = model.to_json()
fn = "../model/model.unet.v109_drop2.json"
print('Writing to file, model : {}'.format(fn))
with open(fn, 'w') as outfile:
    json.dump(json_string, outfile)



