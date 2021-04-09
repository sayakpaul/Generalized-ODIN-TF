# Code is taken from: https://github.com/asmith26/wide_resnets_keras/blob/master/main.py
# Implementation of WideResNet-28-10 (http://arxiv.org/abs/1605.07146)

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Conv2D,
    AveragePooling2D,
    BatchNormalization,
    Dropout,
    Input,
    Activation,
    Add,
    Dense,
    Flatten,
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
import tensorflow as tf

NB_CLASSES = 10
IMAGE_SIZE = 32

WEIGHT_DECAY = 0.0005  # page 10: "Used in all experiments"

DEPTH = 28  # table 5 on page 8 indicates best value (4.17) CIFAR-10
WIDEN_FACTOR = 10  # 'widen_factor'; table 5 on page 8 indicates best value (
# 4.17) CIFAR-10
DROPOUT_PROB = 0  # table 6 on page 10 indicates best value
# (4.17) CIFAR-10

# Other config from code; throughout all layer:
USE_BIAS = (
    False  # following functions 'FCinit(model)' and 'DisableBias(model)' in utils.lua
)
INIT = "he_normal"  # follows the 'MSRinit(model)' function in utils.lua

# Keras specific
if K.image_data_format() == "th":
    channel_axis = 1
    input_shape = (3, IMAGE_SIZE, IMAGE_SIZE)
else:
    channel_axis = -1
    input_shape = (IMAGE_SIZE, IMAGE_SIZE, 3)


# Wide residual network http://arxiv.org/abs/1605.07146
def _wide_basic(n_input_plane, n_output_plane, stride):
    def f(net):
        # format of conv_params:
        #               [ [nb_col="kernel width", nb_row="kernel
        #               height",
        #               subsample="(stride_vertical,
        #               stride_horizontal)",
        #               border_mode="same" or "valid"] ]
        # B(3,3): orignal <<basic>> block
        conv_params = [[3, 3, stride, "same"], [3, 3, (1, 1), "same"]]

        n_bottleneck_plane = n_output_plane

        # Residual block
        for i, v in enumerate(conv_params):
            if i == 0:
                if n_input_plane != n_output_plane:
                    net = BatchNormalization(axis=channel_axis)(net)
                    net = Activation("relu")(net)
                    convs = net
                else:
                    convs = BatchNormalization(axis=channel_axis)(net)
                    convs = Activation("relu")(convs)
                convs = Conv2D(
                    n_bottleneck_plane,
                    (v[0], v[1]),
                    strides=v[2],
                    padding=v[3],
                    kernel_initializer=INIT,
                    kernel_regularizer=l2(WEIGHT_DECAY),
                    use_bias=USE_BIAS,
                )(convs)
            else:
                convs = BatchNormalization(axis=channel_axis)(convs)
                convs = Activation("relu")(convs)
                if DROPOUT_PROB > 0:
                    convs = Dropout(DROPOUT_PROB)(convs)
                convs = Conv2D(
                    n_bottleneck_plane,
                    (v[0], v[1]),
                    strides=v[2],
                    padding=v[3],
                    kernel_initializer=INIT,
                    kernel_regularizer=l2(WEIGHT_DECAY),
                    use_bias=USE_BIAS,
                )(convs)

        # Shortcut Connection: identity function or 1x1
        # convolutional
        #  (depends on difference between input & output shape - this
        #   corresponds to whether we are using the first block in
        #   each
        #   group; see _layer() ).
        if n_input_plane != n_output_plane:
            shortcut = Conv2D(
                n_output_plane,
                (1, 1),
                strides=stride,
                padding="same",
                kernel_initializer=INIT,
                kernel_regularizer=l2(WEIGHT_DECAY),
                use_bias=USE_BIAS,
            )(net)
        else:
            shortcut = net

        return Add()([convs, shortcut])

    return f


# "Stacking Residual Units on the same stage"
def _layer(block, n_input_plane, n_output_plane, count, stride):
    def f(net):
        net = block(n_input_plane, n_output_plane, stride)(net)
        for i in range(2, int(count + 1)):
            net = block(n_output_plane, n_output_plane, stride=(1, 1))(net)
        return net

    return f


def create_model():
    assert (DEPTH - 4) % 6 == 0
    n = (DEPTH - 4) / 6

    inputs = Input(shape=input_shape)
    x = tf.keras.layers.experimental.preprocessing.Rescaling(
        scale=1.0 / 127.5, offset=-1
    )(inputs)
    n_stages = [16, 16 * WIDEN_FACTOR, 32 * WIDEN_FACTOR, 64 * WIDEN_FACTOR]

    conv1 = Conv2D(
        n_stages[0],
        (3, 3),
        strides=1,
        padding="same",
        kernel_initializer=INIT,
        kernel_regularizer=l2(WEIGHT_DECAY),
        use_bias=USE_BIAS,
    )(
        x
    )  # "One conv at the beginning (spatial size: 32x32)"

    # Add wide residual blocks
    block_fn = _wide_basic
    conv2 = _layer(
        block_fn,
        n_input_plane=n_stages[0],
        n_output_plane=n_stages[1],
        count=n,
        stride=(1, 1),
    )(
        conv1
    )  # "Stage 1 (spatial size: 32x32)"
    conv3 = _layer(
        block_fn,
        n_input_plane=n_stages[1],
        n_output_plane=n_stages[2],
        count=n,
        stride=(2, 2),
    )(
        conv2
    )  # "Stage 2 (spatial size: 16x16)"
    conv4 = _layer(
        block_fn,
        n_input_plane=n_stages[2],
        n_output_plane=n_stages[3],
        count=n,
        stride=(2, 2),
    )(
        conv3
    )  # "Stage 3 (spatial size: 8x8)"

    batch_norm = BatchNormalization(axis=channel_axis)(conv4)
    relu = Activation("relu")(batch_norm)

    # Classifier block
    pool = AveragePooling2D(pool_size=(8, 8), strides=(1, 1), padding="same")(relu)
    flatten = Flatten()(pool)

    # Define the ODIN as specified in Section 3.1.1 of
    # https://arxiv.org/abs/2002.11297
    x1 = Dropout(0.7)(flatten)
    h = Dense(NB_CLASSES, kernel_initializer="he_normal")(x1)

    g = Dense(NB_CLASSES, kernel_regularizer=l2(WEIGHT_DECAY))(x1)
    g = BatchNormalization()(g)
    g = Activation("sigmoid")(g)
    outputs = tf.math.divide(h, g)

    model = Model(inputs=inputs, outputs=outputs)
    return model
