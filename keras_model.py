"""
 @file   keras_model.py
 @brief  Script for keras model definition
 @author Toshiki Nakamura, Yuki Nikaido, and Yohei Kawaguchi (Hitachi Ltd.)
 Copyright (C) 2020 Hitachi, Ltd. All right reserved.
"""

########################################################################
# import python-library
########################################################################
# from import
import keras.models
from keras.models import Model
from keras.layers import Input, Dense, BatchNormalization, Activation

########################################################################
# keras model
########################################################################
def get_model(inputDim):
    """
    define the keras model
    the model based on the simple dense auto encoder 
    (128*128*128*128*8*128*128*128*128)
    """
    # inputLayer = Input(shape=(inputDim,))

    # h = Dense(128)(inputLayer)
    # h = BatchNormalization()(h)
    # h = Activation('relu')(h)

    # h = Dense(128)(h)
    # h = BatchNormalization()(h)
    # h = Activation('relu')(h)

    # h = Dense(128)(h)
    # h = BatchNormalization()(h)
    # h = Activation('relu')(h)

    # h = Dense(128)(h)
    # h = BatchNormalization()(h)
    # h = Activation('relu')(h)
    
    # h = Dense(8)(h)
    # h = BatchNormalization()(h)
    # h = Activation('relu')(h)

    # h = Dense(128)(h)
    # h = BatchNormalization()(h)
    # h = Activation('relu')(h)

    # h = Dense(128)(h)
    # h = BatchNormalization()(h)
    # h = Activation('relu')(h)

    # h = Dense(128)(h)
    # h = BatchNormalization()(h)
    # h = Activation('relu')(h)

    # h = Dense(128)(h)
    # h = BatchNormalization()(h)
    # h = Activation('relu')(h)

    # h = Dense(inputDim)(h)

    # Create an input layer with the specified input dimension
    inputLayer = Input(shape=(inputDim,))

    # Add a dense layer with 64 units and ReLU activation function
    h = Dense(64, activation="relu")(inputLayer)

    # Add another dense layer with 64 units and ReLU activation function
    h = Dense(64, activation="relu")(h)

    # Add a dense layer with 8 units and ReLU activation function
    h = Dense(8, activation="relu")(h)

    # Add another dense layer with 64 units and ReLU activation function
    h = Dense(64, activation="relu")(h)

    # Add another dense layer with 64 units and ReLU activation function
    h = Dense(64, activation="relu")(h)

    # Add a dense layer with the same number of units as the input dimension and no activation function
    h = Dense(inputDim, activation=None)(h)

    # Create a model with the input layer as input and the last dense layer as output
    return Model(inputs=inputLayer, outputs=h)
    # return Model(inputs=inputLayer, outputs=h)
#########################################################################


def load_model(file_path):
    return keras.models.load_model(file_path)

    
