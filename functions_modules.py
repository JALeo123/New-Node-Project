from functions_modules import *
from keras_transformer1 import *
import keras
from keras.models import Sequential, Model
from keras.layers.normalization import BatchNormalization
from keras.models import model_from_json
from keras.layers import Input, Dense, Dropout, Activation, Flatten, AveragePooling2D
from keras.layers import Convolution2D, MaxPooling2D, Conv2D
from keras import backend as K
from keras.regularizers import l2
from keras.layers.merge import add
from keras.losses import categorical_crossentropy
from keras.models import model_from_json
#from keras_transformer import get_model, decode, get_custom_objects
import numpy as np
import os
import sys

#Original Designed model
def new_model_original(num_classes_train, num_classes_test, input_shape, add_new_node):
    print("Original Model")
    if add_new_node == False: #Adds softmax layer nodes equal to training classes count
        output_layer = Dense(num_classes_train, activation='softmax')
    else: #Adds extra 'Unknown' node to softmax
        output_layer = Dense(num_classes_train + 1, activation='softmax')

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(output_layer)

    # Default: keras.losses.categorical_crosstentropy
    model.compile(optimizer='adam',
                  metrics=['accuracy'],
                  loss=categorical_crossentropy)
    return model

#ResNet-18 Designed Model, can dynamically make ResNet model bigger
def ResNet_18(num_classes_train, num_classes_test, input_shape, add_new_node):
    print("ResNet-18 Model")
    ROW_AXIS = 1
    COL_AXIS = 2
    CHANNEL_AXIS = 3

    def _bn_relu(input):
        norm = BatchNormalization(axis=CHANNEL_AXIS)(input)
        return Activation("relu")(norm)

    def _conv_bn_relu(**conv_params):
        filters = conv_params["filters"]
        kernel_size = conv_params["kernel_size"]
        strides = conv_params.setdefault("strides", (1, 1))
        kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
        padding = conv_params.setdefault("padding", "same")
        kernel_regularizer = conv_params.setdefault("kernel_regularizer", l2(1.e-4))

        def f(input):
            conv = Conv2D(filters=filters, kernel_size=kernel_size,
                          strides=strides, padding=padding,
                          kernel_initializer=kernel_initializer,
                          kernel_regularizer=kernel_regularizer)(input)
            return _bn_relu(conv)

        return f

    def _bn_relu_conv(**conv_params):
        filters = conv_params["filters"]
        kernel_size = conv_params["kernel_size"]
        strides = conv_params.setdefault("strides", (1, 1))
        kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
        padding = conv_params.setdefault("padding", "same")
        kernel_regularizer = conv_params.setdefault("kernel_regularizer", l2(1.e-4))

        def f(input):
            activation = _bn_relu(input)
            return Conv2D(filters=filters, kernel_size=kernel_size,
                          strides=strides, padding=padding,
                          kernel_initializer=kernel_initializer,
                          kernel_regularizer=kernel_regularizer)(activation)

        return f

    def _shortcut(input, residual):
        # Expand channels of shortcut to match residual.
        # Stride appropriately to match residual (width, height)
        # Should be int if network architecture is correctly configured.
        input_shape = K.int_shape(input)
        residual_shape = K.int_shape(residual)
        stride_width = int(round(input_shape[ROW_AXIS] / residual_shape[ROW_AXIS]))
        stride_height = int(round(input_shape[COL_AXIS] / residual_shape[COL_AXIS]))
        equal_channels = input_shape[CHANNEL_AXIS] == residual_shape[CHANNEL_AXIS]

        shortcut = input
        # 1 X 1 conv if shape is different. Else identity.
        if stride_width > 1 or stride_height > 1 or not equal_channels:
            shortcut = Conv2D(filters=residual_shape[CHANNEL_AXIS],
                              kernel_size=(1, 1),
                              strides=(stride_width, stride_height),
                              padding="valid",
                              kernel_initializer="he_normal",
                              kernel_regularizer=l2(0.0001))(input)

        return add([shortcut, residual])

    def _residual_block(block_function, filters, repetitions, is_first_layer=False):
        def f(input):
            for i in range(repetitions):
                init_strides = (1, 1)
                if i == 0 and not is_first_layer:
                    init_strides = (2, 2)
                input = block_function(filters=filters, init_strides=init_strides,
                                       is_first_block_of_first_layer=(is_first_layer and i == 0))(input)
            return input

        return f

    def basic_block(filters, init_strides=(1, 1), is_first_block_of_first_layer=False):
        def f(input):

            if is_first_block_of_first_layer:
                # don't repeat bn->relu since we just did bn->relu->maxpool
                conv1 = Conv2D(filters=filters, kernel_size=(3, 3),
                               strides=init_strides,
                               padding="same",
                               kernel_initializer="he_normal",
                               kernel_regularizer=l2(1e-4))(input)
            else:
                conv1 = _bn_relu_conv(filters=filters, kernel_size=(3, 3),
                                      strides=init_strides)(input)

            residual = _bn_relu_conv(filters=filters, kernel_size=(3, 3))(conv1)
            return _shortcut(input, residual)

        return f

    input = Input(shape=input_shape)
    conv1 = _conv_bn_relu(filters=64, kernel_size=(7, 7), strides=(2, 2))(input)
    pool1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")(conv1)

    block = pool1
    filters = 64
    for i, r in enumerate([2, 2, 2, 2]):
        block = _residual_block(basic_block, filters=filters, repetitions=r, is_first_layer=(i == 0))(block)
        filters *= 2

    # Last activation
    block = _bn_relu(block)

    # Classifier block
    block_shape = K.int_shape(block)
    pool2 = AveragePooling2D(pool_size=(block_shape[1], block_shape[2]),
                             strides=(1, 1))(block)
    flatten1 = Flatten()(pool2)

    if add_new_node == False:
        dense = Dense(units=num_classes_train, kernel_initializer="he_normal",
                  activation="softmax")(flatten1)
    else:
        dense = Dense(units=num_classes_train + 1, kernel_initializer="he_normal",
                  activation="softmax")(flatten1)

    model = Model(inputs=input, outputs=dense)

    model.compile(optimizer='adam',
                  metrics=['accuracy'],
                  loss=categorical_crossentropy)

    return model

#LeNet-5 Designed Model
def LeNet_5(num_classes_train, num_classes_test, input_shape, add_new_node):
    print("LeNew-5 Model")
    if add_new_node == False: #Adds softmax layer nodes equal to training classes count
        output_layer = Dense(num_classes_train, activation='softmax')
    else: #Adds extra 'Unknown' node to softmax
        output_layer = Dense(num_classes_train + 1, activation='softmax')

    model = keras.Sequential()
    model.add(Conv2D(filters=6, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(AveragePooling2D())
    model.add(Conv2D(filters=16, kernel_size=(3, 3), activation='relu'))
    model.add(AveragePooling2D())
    model.add(Flatten())
    model.add(Dense(units=120, activation='relu'))
    model.add(Dense(units=84, activation='relu'))
    model.add(output_layer)

    # Default: keras.losses.categorical_crosstentropy
    model.compile(optimizer='adam',
                  metrics=['accuracy'],
                  loss=categorical_crossentropy)
    return model

#Transformer Designed Model
def Transformer(num_classes_train, num_classes_test, input_shape, add_new_node, source_token_dict, target_token_dict):
    model = createTransformerModel(num_classes_train, num_classes_test, input_shape, add_new_node, source_token_dict, target_token_dict)

    model.compile(optimizer='adam',
                  metrics=['accuracy'],
                  loss='sparse_categorical_crossentropy')

    return model

def custom_loss_function(y_true, y_pred, from_logits=False, label_smoothing=0): #Categorical_Crossentropy
    y_pred = K.constant(y_pred) if not K.is_tensor(y_pred) else y_pred
    y_true = K.cast(y_true, y_pred.dtype)

    if label_smoothing is not 0:
        smoothing = K.cast_to_floatx(label_smoothing)

        def _smooth_labels():
            num_classes = K.cast(K.shape(y_true)[1], y_pred.dtype)
            return y_true * (1.0 - smoothing) + (smoothing / num_classes)

        y_true = K.switch(K.greater(smoothing, 0), _smooth_labels, lambda: y_true)
    return K.categorical_crossentropy(y_true, y_pred, from_logits=from_logits)