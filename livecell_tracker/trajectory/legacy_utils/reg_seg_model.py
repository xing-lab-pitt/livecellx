import sys

import cv2
import numpy as np
import pandas as pd
from keras import optimizers
from keras.layers import Activation, BatchNormalization, Conv2D, Dropout, Input, MaxPooling2D, UpSampling2D
from keras.models import Model


def conv_block(input_tensor, kernel, filters):
    x = Conv2D(filters, (kernel, kernel), padding="same")(input_tensor)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x


# In[8]:


def reg_seg():
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

    return autoencoder
