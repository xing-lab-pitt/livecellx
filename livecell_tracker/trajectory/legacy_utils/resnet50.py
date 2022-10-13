"""ResNet50 model for Keras.
# Reference:
- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
Adapted from code contributed by BigMoyan.
"""
from __future__ import absolute_import, print_function

import warnings

from keras import backend as K
from keras import layers
from keras.layers import (
    Activation,
    AveragePooling2D,
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    GlobalAveragePooling2D,
    GlobalMaxPooling2D,
    Input,
    MaxPooling2D,
)
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.utils import layer_utils
from keras.utils.data_utils import get_file


def identity_block(input_tensor, kernel_size, filters, stage, block):
    """The identity block is the block that has no conv layer at shortcut.
    # Arguments
            input_tensor: input tensor
            kernel_size: default 3, the kernel size of middle conv layer at main path
            filters: list of integers, the filters of 3 conv layer at main path
            stage: integer, current stage label, used for generating layer names
            block: 'a','b'.., current block label, used for generating layer names
    # Returns
            Output tensor for the block.
    """
    filters1, filters2, filters3 = filters
    if K.image_data_format() == "channels_last":
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = "res" + str(stage) + block + "_branch"
    bn_name_base = "bn" + str(stage) + block + "_branch"

    x = Conv2D(filters1, (1, 1), name=conv_name_base + "2a")(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + "2a")(x)
    x = Activation("relu")(x)

    x = Conv2D(filters2, kernel_size, padding="same", name=conv_name_base + "2b")(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + "2b")(x)
    x = Activation("relu")(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + "2c")(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + "2c")(x)

    x = layers.add([x, input_tensor])
    x = Activation("relu")(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    """A block that has a conv layer at shortcut.
    # Arguments
            input_tensor: input tensor
            kernel_size: default 3, the kernel size of middle conv layer at main path
            filters: list of integers, the filters of 3 conv layer at main path
            stage: integer, current stage label, used for generating layer names
            block: 'a','b'.., current block label, used for generating layer names
    # Returns
            Output tensor for the block.
    Note that from stage 3, the first conv layer at main path is with strides=(2,2)
    And the shortcut should have strides=(2,2) as well
    """
    filters1, filters2, filters3 = filters
    if K.image_data_format() == "channels_last":
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = "res" + str(stage) + block + "_branch"
    bn_name_base = "bn" + str(stage) + block + "_branch"

    x = Conv2D(filters1, (1, 1), strides=strides, name=conv_name_base + "2a")(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + "2a")(x)
    x = Activation("relu")(x)

    x = Conv2D(filters2, kernel_size, padding="same", name=conv_name_base + "2b")(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + "2b")(x)
    x = Activation("relu")(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + "2c")(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + "2c")(x)

    shortcut = Conv2D(filters3, (1, 1), strides=strides, name=conv_name_base + "1")(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + "1")(shortcut)

    x = layers.add([x, shortcut])
    x = Activation("relu")(x)
    return x


def res_model(input_shape, nb_class):
    img_input = Input(input_shape)
    x = Conv2D(64, (7, 7), strides=(2, 2), padding="same", name="conv1")(img_input)
    x = BatchNormalization(axis=3, name="bn_conv1")(x)
    x = Activation("relu")(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block="a", strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block="b")
    x = identity_block(x, 3, [64, 64, 256], stage=2, block="c")

    x = conv_block(x, 3, [128, 128, 512], stage=3, block="a")
    x = identity_block(x, 3, [128, 128, 512], stage=3, block="b")
    x = identity_block(x, 3, [128, 128, 512], stage=3, block="c")
    x = identity_block(x, 3, [128, 128, 512], stage=3, block="d")

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block="a")
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block="b")
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block="c")
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block="d")
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block="e")
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block="f")

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block="a")
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block="b")
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block="c")

    x = AveragePooling2D((2, 2), name="avg_pool")(x)

    x = Flatten()(x)
    x = Dropout(0.5)(x)
    x = Dense(nb_class, activation="softmax")(x)

    model = Model(img_input, x, name="resnet50")
    # model.summary()
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    return model
