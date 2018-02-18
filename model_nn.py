"""
Class for model definitions.
"""

import keras.backend as K

from keras.models import Sequential
from keras import regularizers
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.layers import Lambda
from keras.models import Model
from keras.constraints import maxnorm
from keras.layers import Input
from keras.layers import Dropout
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import MaxPooling1D
from keras.layers.core import Activation, Dense, Flatten
from keras.layers.normalization import BatchNormalization

from utils import pprint as pprint


def model_A(input_shape=None, output_shape=None):
    """
    Returns compiled model.  Defaults to None for all parameters so that errors are thrown if unexpected parameter
    given.

    :return:                BaseModel instance with compiled keras model
    """

    pprint(input_shape)
    pprint(output_shape)

    input = Input(shape=input_shape, name='input')

    # x = Conv1D(filters=128,
    #            kernel_size=39,
    #            activation='relu')(input)
    # x = Conv1D(filters=128,
    #            kernel_size=39,
    #            activation='relu')(x)
    # x = Conv1D(filters=128,
    #            kernel_size=39,
    #            activation='relu')(x)
    # x = Conv1D(filters=128,
    #            kernel_size=39,
    #            activation='relu')(x)
    x = Flatten()(input)
    x = Dense(output_shape[1], input_dim=input_shape[1], activation='relu')(x)
    preds = Dense(output_shape[1], activation='softmax')(x)

    model = Model(input, preds)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adamax',
                  metrics=['categorical_accuracy', 'accuracy'])

    return model


def model_B(input_shape=None, output_shape=None):
    pprint(input_shape)
    pprint(output_shape)

    model = Sequential()
    model.add(Dense(output_shape[1], input_dim=input_shape[1]))
    model.add(Activation("relu"))
    model.add(Dense(output_shape[1]))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adamax',
                  metrics=['accuracy'])

    return model


def model_C(input_shape=None, output_shape=None):
    pprint("Expected input shape: {}".format(input_shape))
    pprint("Expected output shape: {}".format(output_shape))

    model = Sequential()
    model.add(Dense(output_shape[1],
                    input_dim=input_shape[1],
                    kernel_initializer = 'glorot_uniform',
                    kernel_regularizer = regularizers.l2(0),
                    kernel_constraint = maxnorm(9e999)))
    model.add(Activation("relu"))

    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(output_shape[1] * 2,
                    kernel_initializer='glorot_uniform',
                    kernel_regularizer=regularizers.l2(0),
                    kernel_constraint=maxnorm(9e999)))
    model.add(Activation("relu"))

    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(output_shape[1] * 2,
                    kernel_initializer='glorot_uniform',
                    kernel_regularizer=regularizers.l2(0),
                    kernel_constraint=maxnorm(9e999)))
    model.add(Activation("relu"))

    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    # model.add(Dense(output_shape[1] * 2,
    #                 kernel_initializer='glorot_uniform',
    #                 kernel_regularizer=regularizers.l2(0),
    #                 kernel_constraint=maxnorm(9e999)))
    # model.add(Activation("relu"))
    #
    # model.add(BatchNormalization())
    # model.add(Dropout(0.5))

    model.add(Dense(output_shape[1],
                    kernel_initializer='glorot_uniform',
                    kernel_regularizer=regularizers.l2(0),
                    kernel_constraint=maxnorm(9e999)))
    model.add(Activation('softmax'))

    optimus = Adam(lr=0.0001, decay=1e-5)

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimus,
                  metrics=['accuracy'])

    return model