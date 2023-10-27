import os

import keras.backend as K
import numpy as np
import skimage.io as io
import skimage.transform as trans
from keras.applications.resnet50 import ResNet50
from keras.layers import *
from keras.models import *
from keras.optimizers import *

"""
Including several models:
    - res_net
    - res_seg
    - unet
"""

# Keras
# https://www.kaggle.com/bigironsphere/loss-function-library-keras-pytorch
def DiceLoss(targets, inputs, smooth=1e-6):

    # flatten label and prediction tensors
    inputs = K.flatten(inputs)
    targets = K.flatten(targets)

    intersection = K.sum(K.dot(targets, inputs))
    dice = (2 * intersection + smooth) / (K.sum(targets) + K.sum(inputs) + smooth)
    return 1 - dice


def res_net(output_shape):
    """
    ResNet50 is a convolutional neural network that is 50 layers deep.
    """
    model = ResNet50(include_top=False)

    model = Dense(np.prod(output_shape))(model)
    model = Reshape(output_shape)(model)
    model.summary()
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    return model


def conv_block(input_tensor, kernel, filters):
    x = Conv2D(filters, (kernel, kernel), padding="same")(
        input_tensor
    )  # filters: Integer, the dimensionality of the output space
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x


def reg_seg(pretrained_weights=None):
    """
    Defineing CNN
    """
    kernel = 3
    # ------------encoder layers--------------------------------
    inputs = Input((None, None, 1))
    conv1 = conv_block(inputs, kernel, filters=64)
    conv1 = conv_block(conv1, kernel, filters=64)
    pool1 = MaxPooling2D()(conv1)

    conv2 = conv_block(pool1, kernel, filters=128)
    conv2 = conv_block(conv2, kernel, filters=128)
    pool2 = MaxPooling2D()(conv2)

    conv3 = conv_block(pool2, kernel, filters=256)
    conv3 = conv_block(conv3, kernel, filters=256)
    conv3 = conv_block(conv3, kernel, filters=256)
    pool3 = MaxPooling2D()(conv3)

    conv4 = conv_block(pool3, kernel, filters=512)
    conv4 = conv_block(conv4, kernel, filters=512)
    conv4 = conv_block(conv4, kernel, filters=512)
    pool4 = MaxPooling2D()(conv4)

    conv5 = conv_block(pool4, kernel, filters=512)
    conv5 = conv_block(conv5, kernel, filters=512)
    conv5 = conv_block(conv5, kernel, filters=512)
    pool5 = MaxPooling2D()(conv5)

    # --------------------decoder layers--------------------------

    up6 = UpSampling2D()(pool5)
    conv6 = conv_block(up6, kernel, filters=512)
    conv6 = conv_block(conv6, kernel, filters=512)
    conv6 = conv_block(conv6, kernel, filters=512)

    up7 = UpSampling2D()(conv6)
    conv7 = conv_block(up7, kernel, filters=512)
    conv7 = conv_block(conv7, kernel, filters=512)
    conv7 = conv_block(conv7, kernel, filters=512)

    up8 = UpSampling2D()(conv7)
    conv8 = conv_block(up8, kernel, filters=256)
    conv8 = conv_block(conv8, kernel, filters=256)
    conv8 = conv_block(conv8, kernel, filters=256)

    up9 = UpSampling2D()(conv8)
    conv9 = conv_block(up9, kernel, filters=128)
    conv9 = conv_block(conv9, kernel, filters=128)

    up10 = UpSampling2D()(conv9)
    conv10 = conv_block(up10, kernel, filters=64)

    conv11 = conv_block(conv10, kernel=1, filters=1)
    drop11 = Dropout(0.5)(conv11)
    outputs = Activation("relu")(drop11)
    autoencoder = Model(inputs=[inputs], outputs=[outputs])
    # autoencoder.summary()

    autoencoder.compile(loss="mean_squared_error", optimizer="adam", metrics=["mae", "acc"])

    if pretrained_weights:
        autoencoder.load_weights(pretrained_weights)

    return autoencoder


def unet(pretrained_weights=None, input_size=(None, None, 1)):
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation="relu", padding="same", kernel_initializer="he_normal")(inputs)
    conv1 = Conv2D(64, 3, activation="relu", padding="same", kernel_initializer="he_normal")(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation="relu", padding="same", kernel_initializer="he_normal")(pool1)
    conv2 = Conv2D(128, 3, activation="relu", padding="same", kernel_initializer="he_normal")(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation="relu", padding="same", kernel_initializer="he_normal")(pool2)
    conv3 = Conv2D(256, 3, activation="relu", padding="same", kernel_initializer="he_normal")(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation="relu", padding="same", kernel_initializer="he_normal")(pool3)
    conv4 = Conv2D(512, 3, activation="relu", padding="same", kernel_initializer="he_normal")(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation="relu", padding="same", kernel_initializer="he_normal")(pool4)
    conv5 = Conv2D(1024, 3, activation="relu", padding="same", kernel_initializer="he_normal")(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation="relu", padding="same", kernel_initializer="he_normal")(
        UpSampling2D(size=(2, 2))(drop5)
    )
    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = Conv2D(512, 3, activation="relu", padding="same", kernel_initializer="he_normal")(merge6)
    conv6 = Conv2D(512, 3, activation="relu", padding="same", kernel_initializer="he_normal")(conv6)

    up7 = Conv2D(256, 2, activation="relu", padding="same", kernel_initializer="he_normal")(
        UpSampling2D(size=(2, 2))(conv6)
    )
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(256, 3, activation="relu", padding="same", kernel_initializer="he_normal")(merge7)
    conv7 = Conv2D(256, 3, activation="relu", padding="same", kernel_initializer="he_normal")(conv7)

    up8 = Conv2D(128, 2, activation="relu", padding="same", kernel_initializer="he_normal")(
        UpSampling2D(size=(2, 2))(conv7)
    )
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(128, 3, activation="relu", padding="same", kernel_initializer="he_normal")(merge8)
    conv8 = Conv2D(128, 3, activation="relu", padding="same", kernel_initializer="he_normal")(conv8)

    up9 = Conv2D(64, 2, activation="relu", padding="same", kernel_initializer="he_normal")(
        UpSampling2D(size=(2, 2))(conv8)
    )
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(64, 3, activation="relu", padding="same", kernel_initializer="he_normal")(merge9)
    conv9 = Conv2D(64, 3, activation="relu", padding="same", kernel_initializer="he_normal")(conv9)
    conv9 = Conv2D(2, 3, activation="relu", padding="same", kernel_initializer="he_normal")(conv9)
    conv10 = Conv2D(1, 1, activation="sigmoid")(conv9)

    model = Model(inputs=inputs, outputs=conv10)

    model.compile(optimizer=Adam(lr=1e-4), loss=DiceLoss(conv10, inputs, smooth=1e-6), metrics=["mae", "acc"])

    # model.summary()

    if pretrained_weights:
        model.load_weights(pretrained_weights)

    return model
