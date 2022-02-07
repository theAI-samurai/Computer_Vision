"""Keras implementation of SSD."""

from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import ZeroPadding2D
from tensorflow.keras.models import Model

import tensorflow.keras.backend as K
from tensorflow.keras import initializers
from tensorflow.keras.layers import Layer
import tensorflow as tf
from tensorflow.keras.layers import add
from tensorflow.keras.regularizers import l2

from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import BatchNormalization



# from utils.layers import Normalize
#from ssd_model_dense import dsod300_body, dsod512_body
#from ssd_model_resnet import ssd512_resnet_body

kernel_initializer = 'he_normal'
kernel_regularizer = l2(1.e-4)

class Normalize(Layer):
    """Normalization layer as described in ParseNet paper.

    # Arguments
        scale: Default feature scale.

    # Input shape
        4D tensor with shape: (samples, rows, cols, channels)

    # Output shape
        Same as input

    # References
        http://cs.unc.edu/~wliu/papers/parsenet.pdf

    # TODO
        Add possibility to have one scale for all features.
    """

    def __init__(self, scale=20, **kwargs):
        self.scale = scale
        super(Normalize, self).__init__(**kwargs)

    def build(self, input_shape):
        self.gamma = self.add_weight(name=self.name + '_gamma',
                                     shape=(input_shape[-1],),
                                     initializer=initializers.Constant(self.scale),
                                     trainable=True)
        super(Normalize, self).build(input_shape)

    def call(self, x, mask=None):
        return self.gamma * K.l2_normalize(x, axis=-1)

def _to_tensor(x, dtype):
    x = tf.convert_to_tensor(x)
    if x.dtype != dtype:
        x = tf.cast(x, dtype)
    return x

def leaky_relu(x):
    """Leaky Rectified Linear activation.

    # References
        - [Rectifier Nonlinearities Improve Neural Network Acoustic Models](https://web.stanford.edu/~awni/papers/relu_hybrid_icml2013_final.pdf)
    """
    # return K.relu(x, alpha=0.1, max_value=None)

    # requires less memory than keras implementation
    alpha = 0.1
    zero = _to_tensor(0., x.dtype.base_dtype)
    alpha = _to_tensor(alpha, x.dtype.base_dtype)
    x = alpha * tf.minimum(x, zero) + tf.maximum(x, zero)
    return x

def bn_acti_conv(x, filters, kernel_size=1, stride=1, padding='same', activation='relu'):
    x = BatchNormalization(scale=True)(x)
    x = Activation(activation)(x)
    x = Conv2D(filters, kernel_size, strides=stride, padding=padding)(x)
    return x

def _bn_relu_conv(x, filters, kernel_size, strides=(1,1), padding="same"):
    x = BatchNormalization(axis=3)(x)
    x = Activation("relu")(x)
    x = Conv2D(filters, kernel_size, strides=strides, padding=padding,
               kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)(x)
    return x

def bl_bottleneck(x, filters, strides=(1,1), is_first_layer_of_first_block=False):
    if is_first_layer_of_first_block:
        # don't repeat bn->relu since we just did bn->relu->maxpool
        x1 = Conv2D(filters, (1,1), strides=strides, padding="same",
                    kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)(x)
    else:
        x1 = _bn_relu_conv(x, filters=filters, kernel_size=(1,1), strides=strides)
    x1 = _bn_relu_conv(x1, filters=filters, kernel_size=(3,3))
    x1 = _bn_relu_conv(x1, filters=filters*4, kernel_size=(1,1))
    return _shortcut(x, x1)

def _shortcut(x, residual):
    """Adds a shortcut between input and residual block and merges them with "sum"
    """
    # Expand channels of shortcut to match residual.
    # Stride appropriately to match residual (width, height)
    # Should be int if network architecture is correctly configured.
    input_shape = K.int_shape(x)
    residual_shape = K.int_shape(residual)
    stride_width = int(round(input_shape[1] / residual_shape[1]))
    stride_height = int(round(input_shape[2] / residual_shape[2]))
    equal_channels = input_shape[3] == residual_shape[3]

    # 1 X 1 conv if shape is different. Else identity.
    if stride_width > 1 or stride_height > 1 or not equal_channels:
        shortcut = Conv2D(filters=residual_shape[3], kernel_size=(1,1),
                          strides=(stride_width, stride_height), padding="valid",
                          kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)(x)
    else:
        shortcut = x
    return add([shortcut, residual])

def dense_block(x, n, growth_rate, width=4, activation='relu'):
    input_shape = K.int_shape(x)
    c = input_shape[3]
    for i in range(n):
        x1 = x
        x2 = bn_acti_conv(x, growth_rate*width, 1, 1, activation=activation)
        x2 = bn_acti_conv(x2, growth_rate, 3, 1, activation=activation)
        x = concatenate([x1, x2], axis=3)
        c += growth_rate
    return x

def downsampling_block(x, filters, width, padding='same', activation='relu'):
    x1 = MaxPooling2D(pool_size=2, strides=2, padding=padding)(x)
    x1 = bn_acti_conv(x1, filters, 1, 1, padding, activation=activation)
    x2 = bn_acti_conv(x, filters*width, 1, 1, padding, activation=activation)
    x2 = bn_acti_conv(x2, filters, 3, 2, padding, activation=activation)
    return concatenate([x1, x2], axis=3)

def ssd300_body(x):
    source_layers = []

    # Block 1
    x = Conv2D(64, 3, strides=1, padding='same', name='conv1_1', activation='relu')(x)
    x = Conv2D(64, 3, strides=1, padding='same', name='conv1_2', activation='relu')(x)
    x = MaxPool2D(pool_size=2, strides=2, padding='same', name='pool1')(x)
    # Block 2
    x = Conv2D(128, 3, strides=1, padding='same', name='conv2_1', activation='relu')(x)
    x = Conv2D(128, 3, strides=1, padding='same', name='conv2_2', activation='relu')(x)
    x = MaxPool2D(pool_size=2, strides=2, padding='same', name='pool2')(x)
    # Block 3
    x = Conv2D(256, 3, strides=1, padding='same', name='conv3_1', activation='relu')(x)
    x = Conv2D(256, 3, strides=1, padding='same', name='conv3_2', activation='relu')(x)
    x = Conv2D(256, 3, strides=1, padding='same', name='conv3_3', activation='relu')(x)
    x = MaxPool2D(pool_size=2, strides=2, padding='same', name='pool3')(x)
    # Block 4
    x = Conv2D(512, 3, strides=1, padding='same', name='conv4_1', activation='relu')(x)
    x = Conv2D(512, 3, strides=1, padding='same', name='conv4_2', activation='relu')(x)
    x = Conv2D(512, 3, strides=1, padding='same', name='conv4_3', activation='relu')(x)
    source_layers.append(x)
    x = MaxPool2D(pool_size=2, strides=2, padding='same', name='pool4')(x)
    # Block 5
    x = Conv2D(512, 3, strides=1, padding='same', name='conv5_1', activation='relu')(x)
    x = Conv2D(512, 3, strides=1, padding='same', name='conv5_2', activation='relu')(x)
    x = Conv2D(512, 3, strides=1, padding='same', name='conv5_3', activation='relu')(x)
    x = MaxPool2D(pool_size=3, strides=1, padding='same', name='pool5')(x)
    # FC6
    x = Conv2D(1024, 3, strides=1, dilation_rate=(6, 6), padding='same', name='fc6', activation='relu')(x)
    # FC7
    x = Conv2D(1024, 1, strides=1, padding='same', name='fc7', activation='relu')(x)
    source_layers.append(x)
    # Block 6
    x = Conv2D(256, 1, strides=1, padding='same', name='conv6_1', activation='relu')(x)
    x = ZeroPadding2D((1, 1))(x)
    x = Conv2D(512, 3, strides=2, padding='valid', name='conv6_2', activation='relu')(x)
    source_layers.append(x)
    # Block 7
    x = Conv2D(128, 1, strides=1, padding='same', name='conv7_1', activation='relu')(x)
    x = ZeroPadding2D((1, 1))(x)
    x = Conv2D(256, 3, strides=2, padding='valid', name='conv7_2', activation='relu')(x)
    source_layers.append(x)
    # Block 8
    x = Conv2D(128, 1, strides=1, padding='same', name='conv8_1', activation='relu')(x)
    x = Conv2D(256, 3, strides=1, padding='valid', name='conv8_2', activation='relu')(x)
    source_layers.append(x)
    # Block 9
    x = Conv2D(128, 1, strides=1, padding='same', name='conv9_1', activation='relu')(x)
    x = Conv2D(256, 3, strides=1, padding='valid', name='conv9_2', activation='relu')(x)
    source_layers.append(x)

    return source_layers

def ssd512_body(x):
    source_layers = []

    # Block 1
    x = Conv2D(64, 3, strides=1, padding='same', name='conv1_1', activation='relu')(x)
    x = Conv2D(64, 3, strides=1, padding='same', name='conv1_2', activation='relu')(x)
    x = MaxPool2D(pool_size=2, strides=2, padding='same', name='pool1')(x)
    # Block 2
    x = Conv2D(128, 3, strides=1, padding='same', name='conv2_1', activation='relu')(x)
    x = Conv2D(128, 3, strides=1, padding='same', name='conv2_2', activation='relu')(x)
    x = MaxPool2D(pool_size=2, strides=2, padding='same', name='pool2')(x)
    # Block 3
    x = Conv2D(256, 3, strides=1, padding='same', name='conv3_1', activation='relu')(x)
    x = Conv2D(256, 3, strides=1, padding='same', name='conv3_2', activation='relu')(x)
    x = Conv2D(256, 3, strides=1, padding='same', name='conv3_3', activation='relu')(x)
    x = MaxPool2D(pool_size=2, strides=2, padding='same', name='pool3')(x)
    # Block 4
    x = Conv2D(512, 3, strides=1, padding='same', name='conv4_1', activation='relu')(x)
    x = Conv2D(512, 3, strides=1, padding='same', name='conv4_2', activation='relu')(x)
    x = Conv2D(512, 3, strides=1, padding='same', name='conv4_3', activation='relu')(x)
    source_layers.append(x)
    x = MaxPool2D(pool_size=2, strides=2, padding='same', name='pool4')(x)
    # Block 5
    x = Conv2D(512, 3, strides=1, padding='same', name='conv5_1', activation='relu')(x)
    x = Conv2D(512, 3, strides=1, padding='same', name='conv5_2', activation='relu')(x)
    x = Conv2D(512, 3, strides=1, padding='same', name='conv5_3', activation='relu')(x)
    x = MaxPool2D(pool_size=3, strides=1, padding='same', name='pool5')(x)
    # FC6
    x = Conv2D(1024, 3, strides=1, dilation_rate=(6, 6), padding='same', name='fc6', activation='relu')(x)
    # FC7
    x = Conv2D(1024, 1, strides=1, padding='same', name='fc7', activation='relu')(x)
    source_layers.append(x)
    # Block 6
    x = Conv2D(256, 1, strides=1, padding='same', name='conv6_1', activation='relu')(x)
    x = ZeroPadding2D((1, 1))(x)
    x = Conv2D(512, 3, strides=2, padding='valid', name='conv6_2', activation='relu')(x)
    source_layers.append(x)
    # Block 7
    x = Conv2D(128, 1, strides=1, padding='same', name='conv7_1', activation='relu')(x)
    x = ZeroPadding2D((1, 1))(x)
    x = Conv2D(256, 3, strides=2, padding='valid', name='conv7_2', activation='relu')(x)
    source_layers.append(x)
    # Block 8
    x = Conv2D(128, 1, strides=1, padding='same', name='conv8_1', activation='relu')(x)
    x = ZeroPadding2D((1, 1))(x)
    x = Conv2D(256, 3, strides=2, padding='valid', name='conv8_2', activation='relu')(x)
    source_layers.append(x)
    # Block 9
    x = Conv2D(128, 1, strides=1, padding='same', name='conv9_1', activation='relu')(x)
    x = ZeroPadding2D((1, 1))(x)
    x = Conv2D(256, 3, strides=2, padding='valid', name='conv9_2', activation='relu')(x)
    source_layers.append(x)
    # Block 10
    x = Conv2D(128, 1, strides=1, padding='same', name='conv10_1', activation='relu')(x)
    x = ZeroPadding2D((1, 1))(x)
    x = Conv2D(256, 4, strides=2, padding='valid', name='conv10_2', activation='relu')(x)
    source_layers.append(x)

    return source_layers

def dsod300_body(x, activation='relu'):
    if activation == 'leaky_relu':
        activation = leaky_relu

    growth_rate = 48
    compression = 1.0
    source_layers = []

    # Stem
    x = Conv2D(64, 3, strides=2, padding='same')(x)
    x = BatchNormalization(scale=True)(x)
    x = Activation(activation)(x)
    x = Conv2D(64, 3, strides=1, padding='same')(x)
    x = BatchNormalization(scale=True)(x)
    x = Activation(activation)(x)
    x = Conv2D(128, 3, strides=1, padding='same')(x)
    x = BatchNormalization(scale=True)(x)
    x = Activation(activation)(x)
    x = MaxPooling2D(pool_size=2, strides=2)(x)
    # Dense Block 1
    x = dense_block(x, 6, growth_rate, 4, activation)
    x = bn_acti_conv(x, int(K.int_shape(x)[3] * compression), 1, 1, activation=activation)
    x = MaxPooling2D(pool_size=2, strides=2, padding='same')(x)
    # Dense Block 2
    x = dense_block(x, 8, growth_rate, 4, activation)
    x = bn_acti_conv(x, int(K.int_shape(x)[3] * compression), 1, 1, activation=activation)
    source_layers.append(x)  # 38x38

    x = MaxPooling2D(pool_size=2, strides=2)(x)
    x2 = x
    # Dense Block 3
    x = dense_block(x, 8, growth_rate, 4, activation)
    x = bn_acti_conv(x, int(K.int_shape(x)[3] * compression), 1, 1, activation=activation)
    # Dense Block 4
    x = dense_block(x, 8, growth_rate, 4, activation)
    x1 = x

    x1 = bn_acti_conv(x1, 256, 1, 1, activation=activation)
    x2 = bn_acti_conv(x2, 256, 1, 1, activation=activation)
    x = concatenate([x1, x2], axis=3)
    source_layers.append(x)  # 19x19

    x = downsampling_block(x, 256, 1, activation=activation)
    source_layers.append(x)  # 10x10

    x = downsampling_block(x, 128, 1, activation=activation)
    source_layers.append(x)  # 5x5

    x = downsampling_block(x, 128, 1, activation=activation)
    source_layers.append(x)  # 3x3

    x = downsampling_block(x, 128, 1, padding='valid', activation=activation)
    source_layers.append(x)  # 1x1

    return source_layers

def dsod512_body(x, activation='relu'):
    if activation == 'leaky_relu':
        activation = leaky_relu

    growth_rate = 48
    compression = 1.0
    source_layers = []

    # Stem
    x = Conv2D(64, 3, strides=2, padding='same')(x)
    x = BatchNormalization(scale=True)(x)
    x = Activation(activation)(x)
    x = Conv2D(64, 3, strides=1, padding='same')(x)
    x = BatchNormalization(scale=True)(x)
    x = Activation(activation)(x)
    x = Conv2D(128, 3, strides=1, padding='same')(x)
    x = BatchNormalization(scale=True)(x)
    x = Activation(activation)(x)
    x = MaxPooling2D(pool_size=2, strides=2)(x)

    # Dense Block 1
    x = dense_block(x, 6, growth_rate, 4, activation)
    x = bn_acti_conv(x, int(K.int_shape(x)[3] * compression), 1, 1, activation=activation)
    x = MaxPooling2D(pool_size=2, strides=2, padding='same')(x)
    # Dense Block 2
    x = dense_block(x, 8, growth_rate, 4, activation)
    x = bn_acti_conv(x, int(K.int_shape(x)[3] * compression), 1, 1, activation=activation)
    source_layers.append(x)  # 64x64

    x = MaxPooling2D(pool_size=2, strides=2)(x)
    x2 = x
    # Dense Block 3
    x = dense_block(x, 8, growth_rate, 4, activation)
    x = bn_acti_conv(x, int(K.int_shape(x)[3] * compression), 1, 1, activation=activation)
    # Dense Block 4
    x = dense_block(x, 8, growth_rate, 4, activation)
    x1 = x

    x1 = bn_acti_conv(x1, 256, 1, 1, activation=activation)
    x2 = bn_acti_conv(x2, 256, 1, 1, activation=activation)
    x = concatenate([x1, x2], axis=3)
    source_layers.append(x)  # 32x32

    x = downsampling_block(x, 256, 1, activation=activation)
    source_layers.append(x)  # 16x16

    x = downsampling_block(x, 128, 1, activation=activation)
    source_layers.append(x)  # 8x8

    x = downsampling_block(x, 128, 1, activation=activation)
    source_layers.append(x)  # 4x4

    x = downsampling_block(x, 128, 1, activation=activation)
    source_layers.append(x)  # 2x2

    x = downsampling_block(x, 128, 1, activation=activation)
    source_layers.append(x)  # 1x1

    return source_layers

def ssd512_resnet_body(x, activation='relu'):
    if activation == 'leaky_relu':
        activation = leaky_relu

    source_layers = []

    x = Conv2D(64, (7, 7), strides=(2, 2), padding='same',
               kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation(activation)(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

    x = bl_bottleneck(x, filters=64, is_first_layer_of_first_block=True)
    x = bl_bottleneck(x, filters=64)
    x = bl_bottleneck(x, filters=64)

    x = bl_bottleneck(x, filters=128, strides=(2, 2))
    x = bl_bottleneck(x, filters=128)
    x = bl_bottleneck(x, filters=128)
    x = bl_bottleneck(x, filters=128)
    source_layers.append(x)

    x = bl_bottleneck(x, filters=256, strides=(2, 2))
    x = bl_bottleneck(x, filters=256)
    x = bl_bottleneck(x, filters=256)
    source_layers.append(x)

    x = bl_bottleneck(x, filters=256, strides=(2, 2))
    x = bl_bottleneck(x, filters=256)
    x = bl_bottleneck(x, filters=256)
    source_layers.append(x)

    x = bl_bottleneck(x, filters=256, strides=(2, 2))
    x = bl_bottleneck(x, filters=256)
    source_layers.append(x)

    x = bl_bottleneck(x, filters=256, strides=(2, 2))
    x = bl_bottleneck(x, filters=256)
    source_layers.append(x)

    x = bl_bottleneck(x, filters=256, strides=(2, 2))
    x = bl_bottleneck(x, filters=256)
    source_layers.append(x)

    x = bl_bottleneck(x, filters=256, strides=(2, 2))
    x = bl_bottleneck(x, filters=256)
    source_layers.append(x)

    return source_layers

def multibox_head(source_layers, num_priors, num_classes, normalizations=None, softmax=True):
    class_activation = 'softmax' if softmax else 'sigmoid'

    mbox_conf = []
    mbox_loc = []
    for i in range(len(source_layers)):
        x = source_layers[i]
        name = x.name.split('/')[0]

        # normalize
        if normalizations is not None and normalizations[i] > 0:
            name = name + '_norm'
            x = Normalize(normalizations[i], name=name)(x)

        # confidence
        name1 = name + '_mbox_conf'
        x1 = Conv2D(num_priors[i] * num_classes, 3, padding='same', name=name1)(x)
        x1 = Flatten(name=name1 + '_flat')(x1)
        mbox_conf.append(x1)

        # location
        name2 = name + '_mbox_loc'
        x2 = Conv2D(num_priors[i] * 4, 3, padding='same', name=name2)(x)
        x2 = Flatten(name=name2 + '_flat')(x2)
        mbox_loc.append(x2)

    mbox_loc = concatenate(mbox_loc, axis=1, name='mbox_loc')
    mbox_loc = Reshape((-1, 4), name='mbox_loc_final')(mbox_loc)

    mbox_conf = concatenate(mbox_conf, axis=1, name='mbox_conf')
    mbox_conf = Reshape((-1, num_classes), name='mbox_conf_logits')(mbox_conf)
    mbox_conf = Activation(class_activation, name='mbox_conf_final')(mbox_conf)

    predictions = concatenate([mbox_loc, mbox_conf], axis=2, name='predictions')

    return predictions

def SSD300(input_shape=(300, 300, 3), num_classes=21, softmax=True):
    """SSD300 architecture.

    # Arguments
        input_shape: Shape of the input image.
        num_classes: Number of classes including background.

    # Notes
        In order to stay compatible with pre-trained models, the parameters
        were chosen as in the caffee implementation.

    # References
        https://arxiv.org/abs/1512.02325
    """
    x = input_tensor = Input(shape=input_shape)
    source_layers = ssd300_body(x)

    # Add multibox head for classification and regression
    num_priors = [4, 6, 6, 6, 4, 4]
    normalizations = [20, -1, -1, -1, -1, -1]
    output_tensor = multibox_head(source_layers, num_priors, num_classes, normalizations, softmax)
    model = Model(input_tensor, output_tensor)
    model.num_classes = num_classes

    # parameters for prior boxes
    model.image_size = input_shape[:2]
    model.source_layers = source_layers
    # stay compatible with caffe models
    model.aspect_ratios = [[1, 2, 1 / 2], [1, 2, 1 / 2, 3, 1 / 3], [1, 2, 1 / 2, 3, 1 / 3], [1, 2, 1 / 2, 3, 1 / 3],
                           [1, 2, 1 / 2], [1, 2, 1 / 2]]
    model.minmax_sizes = [(30, 60), (60, 111), (111, 162), (162, 213), (213, 264), (264, 315)]
    model.steps = [8, 16, 32, 64, 100, 300]
    model.special_ssd_boxes = True

    return model

def SSD512(input_shape=(512, 512, 3), num_classes=21, softmax=True):
    """SSD512 architecture.

    # Arguments
        input_shape: Shape of the input image.
        num_classes: Number of classes including background.

    # Notes
        In order to stay compatible with pre-trained models, the parameters
        were chosen as in the caffee implementation.

    # References
        https://arxiv.org/abs/1512.02325
    """
    x = input_tensor = Input(shape=input_shape)
    source_layers = ssd512_body(x)

    # Add multibox head for classification and regression
    num_priors = [4, 6, 6, 6, 6, 4, 4]
    normalizations = [20, -1, -1, -1, -1, -1, -1]
    output_tensor = multibox_head(source_layers, num_priors, num_classes, normalizations, softmax)
    model = Model(input_tensor, output_tensor)
    model.num_classes = num_classes

    # parameters for prior boxes
    model.image_size = input_shape[:2]
    model.source_layers = source_layers
    # stay compatible with caffe models
    model.aspect_ratios = [[1, 2, 1 / 2], [1, 2, 1 / 2, 3, 1 / 3], [1, 2, 1 / 2, 3, 1 / 3], [1, 2, 1 / 2, 3, 1 / 3],
                           [1, 2, 1 / 2, 3, 1 / 3], [1, 2, 1 / 2], [1, 2, 1 / 2]]
    # model.minmax_sizes = [(35, 76), (76, 153), (153, 230), (230, 307), (307, 384), (384, 460), (460, 537)]
    model.minmax_sizes = [(20.48, 51.2), (51.2, 133.12), (133.12, 215.04), (215.04, 296.96), (296.96, 378.88),
                          (378.88, 460.8), (460.8, 542.72)]
    model.steps = [8, 16, 32, 64, 128, 256, 512]
    model.special_ssd_boxes = True

    return model

def ssd384x512_dense_body(x, activation='relu'):
    # used for SegLink 384x512

    if activation == 'leaky_relu':
        activation = leaky_relu

    growth_rate = 32
    compression = 1.0
    source_layers = []

    # Stem
    x = Conv2D(64, 3, strides=2, padding='same')(x)
    x = BatchNormalization(scale=True)(x)
    x = Activation(activation)(x)
    x = Conv2D(64, 3, strides=1, padding='same')(x)
    x = BatchNormalization(scale=True)(x)
    x = Activation(activation)(x)
    x = Conv2D(96, 3, strides=1, padding='same')(x)
    x = BatchNormalization(scale=True)(x)
    x = Activation(activation)(x)
    x = MaxPooling2D(pool_size=2, strides=2)(x)

    x = dense_block(x, 6, growth_rate, 4, activation)
    x = bn_acti_conv(x, int(K.int_shape(x)[3] * compression), 1, 1, activation=activation)

    x = dense_block(x, 8, growth_rate, 4, activation)
    x = bn_acti_conv(x, int(K.int_shape(x)[3] * compression), 1, 1, activation=activation)
    source_layers.append(x)

    x = downsampling_block(x, 320, 1, activation=activation)
    source_layers.append(x)

    x = downsampling_block(x, 256, 1, activation=activation)
    source_layers.append(x)

    x = downsampling_block(x, 192, 1, activation=activation)
    source_layers.append(x)

    x = downsampling_block(x, 128, 1, activation=activation)
    source_layers.append(x)

    x = downsampling_block(x, 64, 1, activation=activation)
    source_layers.append(x)

    return source_layers

def DSOD300(input_shape=(300, 300, 3), num_classes=21, activation='relu', softmax=True):
    """DSOD, DenseNet based SSD300 architecture.

    # Arguments
        input_shape: Shape of the input image.
        num_classes: Number of classes including background.
        activation: Type of activation functions.

    # References
        https://arxiv.org/abs/1708.01241
    """
    x = input_tensor = Input(shape=input_shape)
    source_layers = dsod300_body(x, activation=activation)

    num_priors = [4, 6, 6, 6, 4, 4]
    normalizations = [20, 20, 20, 20, 20, 20]

    output_tensor = multibox_head(source_layers, num_priors, num_classes, normalizations, softmax)
    model = Model(input_tensor, output_tensor)
    model.num_classes = num_classes

    # parameters for prior boxes
    model.image_size = input_shape[:2]
    model.source_layers = source_layers
    model.aspect_ratios = [[1, 2, 1 / 2], [1, 2, 1 / 2, 3, 1 / 3], [1, 2, 1 / 2, 3, 1 / 3], [1, 2, 1 / 2, 3, 1 / 3],
                           [1, 2, 1 / 2], [1, 2, 1 / 2]]
    model.minmax_sizes = [(30, 60), (60, 111), (111, 162), (162, 213), (213, 264), (264, 315)]
    model.steps = [8, 16, 32, 64, 100, 300]
    model.special_ssd_boxes = True

    return model


SSD300_dense = DSOD300


def DSOD512(input_shape=(512, 512, 3), num_classes=21, activation='relu', softmax=True):
    """DSOD, DenseNet based SSD512 architecture.

    # Arguments
        input_shape: Shape of the input image.
        num_classes: Number of classes including background.
        activation: Type of activation functions.

    # References
        https://arxiv.org/abs/1708.01241
    """
    x = input_tensor = Input(shape=input_shape)
    source_layers = dsod512_body(x, activation=activation)

    num_priors = [4, 6, 6, 6, 6, 4, 4]
    normalizations = [20, 20, 20, 20, 20, 20, 20]

    output_tensor = multibox_head(source_layers, num_priors, num_classes, normalizations, softmax)
    model = Model(input_tensor, output_tensor)
    model.num_classes = num_classes

    # parameters for prior boxes
    model.image_size = input_shape[:2]
    model.source_layers = source_layers
    model.aspect_ratios = [[1, 2, 1 / 2], [1, 2, 1 / 2, 3, 1 / 3], [1, 2, 1 / 2, 3, 1 / 3], [1, 2, 1 / 2, 3, 1 / 3],
                           [1, 2, 1 / 2, 3, 1 / 3], [1, 2, 1 / 2], [1, 2, 1 / 2]]
    model.minmax_sizes = [(35, 76), (76, 153), (153, 230), (230, 307), (307, 384), (384, 460), (460, 537)]
    model.steps = [8, 16, 32, 64, 128, 256, 512]
    model.special_ssd_boxes = True

    return model


SSD512_dense = DSOD512


def SSD512_resnet(input_shape=(512, 512, 3), num_classes=21, softmax=True):
    # TODO: it does not converge!

    x = input_tensor = Input(shape=input_shape)
    source_layers = ssd512_resnet_body(x)

    # Add multibox head for classification and regression
    num_priors = [4, 6, 6, 6, 6, 4, 4]
    normalizations = [20, 20, 20, 20, 20, 20, 20]
    output_tensor = multibox_head(source_layers, num_priors, num_classes, normalizations, softmax)
    model = Model(input_tensor, output_tensor)
    model.num_classes = num_classes

    # parameters for prior boxes
    model.image_size = input_shape[:2]
    model.source_layers = source_layers
    # stay compatible with caffe models
    model.aspect_ratios = [[1, 2, 1 / 2], [1, 2, 1 / 2, 3, 1 / 3], [1, 2, 1 / 2, 3, 1 / 3], [1, 2, 1 / 2, 3, 1 / 3],
                           [1, 2, 1 / 2, 3, 1 / 3], [1, 2, 1 / 2], [1, 2, 1 / 2]]
    model.minmax_sizes = [(35, 76), (76, 153), (153, 230), (230, 307), (307, 384), (384, 460), (460, 537)]
    model.steps = [8, 16, 32, 64, 128, 256, 512]
    model.special_ssd_boxes = True

    return model
