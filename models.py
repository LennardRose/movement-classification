# ---------------------------------------------------------------------------------------------------#
# File name: models.py                                                                               #
# Created on: 17.12.2022                                                                             #
# ---------------------------------------------------------------------------------------------------#
# Learning of Structured Data (FHWS WS22/23) - Skeleton Data time series classification
# This file provides different models.
# Exact description in the functions.


import numpy as np
from tensorflow.keras.models import Model
import tensorflow.keras.layers as layer
from tensorflow.keras import layers, initializers


def mlp_net(data):
    """This function creates a MLP model in TensorFlow.

    Args:
        data (numpy array): data fed into the model, here only relevant to find out the input shape

    Returns:
        model (TensorFlow / Keras model): model for training and testing
    """

    x_input = layer.Input(shape=(data.shape[-2:]))

    x = layer.Flatten()(x_input)
    x = layer.Dense(256)(x)
    x = layer.BatchNormalization()(x)
    x = layer.ReLU()(x)
    x = layer.Dropout(0.5)(x)

    x = layer.Dense(128)(x)
    x = layer.BatchNormalization()(x)
    x = layer.ReLU()(x)
    x = layer.Dropout(0.5)(x)
    x_output = layer.Dense(5, activation = "softmax")(x)

    model = Model(inputs=x_input, outputs=x_output, name="mlp_net")

    return model


def cnn_net(data):
    """This function creates a CNN model in TensorFlow.

    Args:
        data (numpy array): data fed into the model, here only relevant to find out the input shape

    Returns:
        model (TensorFlow / Keras model): model for training and testing
    """

    x_input = layer.Input(shape=(data.shape[-3:]))

    x = layer.Conv2D(filters = 64, kernel_size = (2, 2))(x_input)
    x = layer.BatchNormalization()(x)
    x = layer.ReLU()(x)
    x = layer.Conv2D(filters = 128, kernel_size = (2, 2))(x)
    x = layer.BatchNormalization()(x)
    x = layer.ReLU()(x)
    x = layer.AveragePooling2D()(x)

    x = layer.Flatten()(x)
    x = layer.Dense(128)(x)
    x = layer.BatchNormalization()(x)
    x = layer.ReLU()(x)
    x = layer.Dropout(0.5)(x)
    x_output = layer.Dense(5, activation = "softmax")(x)

    model = Model(inputs=x_input, outputs=x_output, name="cnn_net")

    return model


def cnn_net_v2(data):
    """This function creates a CNN model version 2 in TensorFlow.

    Args:
        data (numpy array): data fed into the model, here only relevant to find out the input shape

    Returns:
        model (TensorFlow / Keras model): model for training and testing
    """

    x_input = layer.Input(shape=(data.shape[-3:]))

    x = layer.Conv2D(filters = 64, kernel_size = (2, 2))(x_input)
    x = layer.BatchNormalization()(x)
    x = layer.ReLU()(x)
    x = layer.MaxPooling2D(pool_size = (2, 2))(x)
    x = layer.Conv2D(filters = 128, kernel_size = (2, 2))(x)
    x = layer.BatchNormalization()(x)
    x = layer.ReLU()(x)
    x = layer.MaxPooling2D(pool_size = (2, 2))(x)
    x = layer.Conv2D(filters = 256, kernel_size = (2, 2))(x)
    x = layer.BatchNormalization()(x)
    x = layer.ReLU()(x)
    x = layer.AveragePooling2D()(x)

    x = layer.Flatten()(x)
    x = layer.Dense(128)(x)
    x = layer.BatchNormalization()(x)
    x = layer.ReLU()(x)
    x = layer.Dropout(0.5)(x)
    x_output = layer.Dense(5, activation = "softmax")(x)

    model = Model(inputs = x_input, outputs = x_output, name = "cnn_net_v2")

    return model
    

def gru_net(data):
    """This function creates a GRU model in TensorFlow.

    Args:
        data (numpy array): data fed into the model, here only relevant to find out the input shape

    Returns:
        model (TensorFlow / Keras model): model for training and testing
    """

    x_input = layer.Input(shape=(data.shape[-2:]))

    x = layer.Bidirectional(layer.GRU(256, return_sequences = True))(x_input)
    x = layer.BatchNormalization()(x)
    x = layer.Bidirectional(layer.GRU(128, return_sequences = True))(x)
    x = layer.BatchNormalization()(x)

    x = layer.Flatten()(x)
    x = layer.Dense(128)(x)
    x = layer.BatchNormalization()(x)
    x = layer.ReLU()(x)
    x = layer.Dropout(0.5)(x)
    x_output = layer.Dense(5, activation = "softmax")(x)

    model = Model(inputs=x_input, outputs=x_output, name="gru_net")

    return model


def conv_lstm_net(data):
    """This function creates a convolutional LSTM model in TensorFlow.

    Args:
        data (numpy array): data fed into the model, here only relevant to find out the input shape

    Returns:
        model (TensorFlow / Keras model): model for training and testing
    """

    x_input = layer.Input(shape = (data.shape[-4:]))

    # 1. Conv
    # 2. BatchNormalization 
    # 3. Activation
    # 4. Pooling
    # 5. Dropout
    # LSTM become less and less neurons from layer to layer
    # CNN become more and more filters from layer to layer or the same number of filters remain
    x = layer.Bidirectional(layer.ConvLSTM2D(filters = 64, kernel_size = (3, 3), return_sequences = True))(x_input)
    x = layer.BatchNormalization()(x)
    x = layer.Bidirectional(layer.ConvLSTM2D(filters = 128, kernel_size = (3, 3), return_sequences = True))(x)
    x = layer.BatchNormalization()(x)
    x = layer.AveragePooling3D(pool_size = (2, 2, 2))(x)

    x = layer.Flatten()(x)
    x = layer.Dense(128)(x)
    x = layer.BatchNormalization()(x)
    x = layer.ReLU()(x)
    x = layer.Dropout(0.5)(x)
    x_output = layer.Dense(5, activation = "softmax")(x)

    model = Model(inputs = x_input, outputs = x_output, name = "conv_lstm_net")

    return model

    
#---------------------------------------------------------------------------------------------------#
# ResNet18 & ResNet34
# Inspired by: https://github.com/jimmyyhwu/resnet18-tf2 and 
#              https://github.com/pytorch/vision/blob/v0.4.0/torchvision/models/resnet.py


kaiming_normal = initializers.VarianceScaling(scale = 2.0, mode = "fan_out", distribution = "untruncated_normal")


def conv3x3(x, out_planes, stride = 1, name = None):

    x = layers.ZeroPadding2D(padding = 1, name = f'{name}_pad')(x)
    layer = layers.Conv2D(filters = out_planes, kernel_size = 3, strides = stride, use_bias = False, 
                          kernel_initializer = kaiming_normal, name = name)(x)

    return layer


def basic_block(x, planes, stride = 1, downsample = None, name = None):

    identity = x

    out = conv3x3(x, planes, stride = stride, name = f'{name}.conv1')
    out = layers.BatchNormalization(momentum = 0.9, epsilon = 1e-5, name = f'{name}.bn1')(out)
    out = layers.ReLU(name = f'{name}.relu1')(out)

    out = conv3x3(out, planes, name = f'{name}.conv2')
    out = layers.BatchNormalization(momentum = 0.9, epsilon = 1e-5, name = f'{name}.bn2')(out)

    if (downsample is not None):
        for layer in downsample:
            identity = layer(identity)

    out = layers.Add(name = f'{name}.add')([identity, out])
    out = layers.ReLU(name = f'{name}.relu2')(out)

    return out


def create_layer(x, planes, blocks, stride = 1, name = None):

    downsample = None
    inplanes = x.shape[3]

    if ((stride != 1) or (inplanes != planes)):
        downsample = [
            layers.Conv2D(filters = planes, kernel_size = 1, strides = stride, use_bias = False, 
                          kernel_initializer = kaiming_normal, name = f'{name}.0.downsample.0'),
            layers.BatchNormalization(momentum = 0.9, epsilon = 1e-5, name = f'{name}.0.downsample.1'),
        ]

    x = basic_block(x, planes, stride, downsample, name = f'{name}.0')

    for i in range(1, blocks):
        x = basic_block(x, planes, name = f'{name}.{i}')

    return x


def resnet_original(x, blocks_per_layer, num_classes = 1000):

    initializer = initializers.RandomUniform(-1.0 / np.sqrt(512), 1.0 / np.sqrt(512))

    x = layers.ZeroPadding2D(padding = 3, name = "conv1_pad")(x)
    x = layers.Conv2D(filters = 64, kernel_size = 7, strides = 2, use_bias = False, 
                      kernel_initializer = kaiming_normal, name = "conv1")(x)
    x = layers.BatchNormalization(momentum = 0.9, epsilon = 1e-5, name = "bn1")(x)
    x = layers.ReLU(name = "relu1")(x)
    x = layers.ZeroPadding2D(padding = 1, name = "maxpool_pad")(x)
    x = layers.MaxPool2D(pool_size = 3, strides = 2, name = "maxpool")(x)

    x = create_layer(x,  64, blocks_per_layer[0], name = "layer1")
    x = create_layer(x, 128, blocks_per_layer[1], stride = 2, name = "layer2")
    x = create_layer(x, 256, blocks_per_layer[2], stride = 2, name = "layer3")
    x = create_layer(x, 512, blocks_per_layer[3], stride = 2, name = "layer4")

    x = layers.GlobalAveragePooling2D(name = "avgpool")(x)
    x = layers.Dense(units = num_classes, kernel_initializer = initializer, bias_initializer = initializer, 
                     name = "fc")(x)

    return x


def resnet(x, blocks_per_layer, num_classes = 5):

    initializer = initializers.RandomUniform(-1.0 / np.sqrt(512), 1.0 / np.sqrt(512))

    x = layers.ZeroPadding2D(padding = 3, name = "conv1_pad")(x)
    x = layers.Conv2D(filters = 64, kernel_size = 7, strides = 2, use_bias = False, 
                      kernel_initializer = kaiming_normal, name = "conv1")(x)
    x = layers.BatchNormalization(momentum = 0.9, epsilon = 1e-5, name = "bn1")(x)
    x = layers.ReLU(name = "relu1")(x)
    x = layers.ZeroPadding2D(padding = 1, name = "maxpool_pad")(x)
    x = layers.MaxPool2D(pool_size = 3, strides = 2, name = "maxpool")(x)

    x = create_layer(x,  64, blocks_per_layer[0], name = "layer1")
    x = create_layer(x, 128, blocks_per_layer[1], stride = 2, name = "layer2")
    x = create_layer(x, 256, blocks_per_layer[2], stride = 2, name = "layer3")
    x = create_layer(x, 512, blocks_per_layer[3], stride = 2, name = "layer4")

    x = layers.GlobalAveragePooling2D(name = "avgpool")(x)
    x = layers.BatchNormalization(name = "bn_end")(x)
    x = layers.Dropout(0.5, name = "dropout_end")(x)
    x = layers.Dense(units = num_classes, kernel_initializer = initializer, bias_initializer = initializer, 
                     name = "fc", activation = "softmax")(x)

    return x


def resnet18(data, **kwargs):

    x_input = layers.Input(shape = (data.shape[-3:]))
    x_output = resnet(x_input, [2, 2, 2, 2], **kwargs) 

    model = Model(inputs = x_input, outputs = x_output, name = "resnet18")

    return model


def resnet34(data, **kwargs):

    x_input = layers.Input(shape = (data.shape[-3:]))
    x_output = resnet(x_input, [3, 4, 6, 3], **kwargs)

    model = Model(inputs = x_input, outputs = x_output, name = "resnet34")

    return model

