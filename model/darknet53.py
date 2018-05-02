"""Darknet-53 for yolo v3.
"""
from keras.models import Model
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, Dense
from keras.layers import add, Activation, BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.regularizers import l2


def conv2d_unit(x, filters, kernels, strides=1):
    """Convolution Unit
    This function defines a 2D convolution operation with BN and LeakyReLU.

    # Arguments
        x: Tensor, input tensor of conv layer.
        filters: Integer, the dimensionality of the output space.
        kernels: An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window.
        strides: An integer or tuple/list of 2 integers,
            specifying the strides of the convolution along the width and
            height. Can be a single integer to specify the same value for
            all spatial dimensions.

    # Returns
            Output tensor.
    """
    x = Conv2D(filters, kernels,
               padding='same',
               strides=strides,
               activation='linear',
               kernel_regularizer=l2(5e-4))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)

    return x


def residual_block(inputs, filters):
    """Residual Block
    This function defines a 2D convolution operation with BN and LeakyReLU.

    # Arguments
        x: Tensor, input tensor of residual block.
        kernels: An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window.

    # Returns
        Output tensor.
    """
    x = conv2d_unit(inputs, filters, (1, 1))
    x = conv2d_unit(x, 2 * filters, (3, 3))
    x = add([inputs, x])
    x = Activation('linear')(x)

    return x


def stack_residual_block(inputs, filters, n):
    """Stacked residual Block
    """
    x = residual_block(inputs, filters)

    for i in range(n - 1):
        x = residual_block(x, filters)

    return x


def darknet_base(inputs):
    """Darknet-53 base model.
    """

    x = conv2d_unit(inputs, 32, (3, 3))

    x = conv2d_unit(x, 64, (3, 3), strides=2)
    x = stack_residual_block(x, 32, n=1)

    x = conv2d_unit(x, 128, (3, 3), strides=2)
    x = stack_residual_block(x, 64, n=2)

    x = conv2d_unit(x, 256, (3, 3), strides=2)
    x = stack_residual_block(x, 128, n=8)

    x = conv2d_unit(x, 512, (3, 3), strides=2)
    x = stack_residual_block(x, 256, n=8)

    x = conv2d_unit(x, 1024, (3, 3), strides=2)
    x = stack_residual_block(x, 512, n=4)

    return x


def darknet():
    """Darknet-53 classifier.
    """
    inputs = Input(shape=(416, 416, 3))
    x = darknet_base(inputs)

    x = GlobalAveragePooling2D()(x)
    x = Dense(1000, activation='softmax')(x)

    model = Model(inputs, x)

    return model


if __name__ == '__main__':
    model = darknet()
    print(model.summary())
