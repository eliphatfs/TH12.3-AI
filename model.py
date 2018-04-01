# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 11:53:33 2018

@author: 北海若
"""

import keras


def get_model():
    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(32, 6, strides=(2, 2), input_shape=(80, 80, 3)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.LeakyReLU())
    model.add(keras.layers.Conv2D(64, 6, strides=(2, 2)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.LeakyReLU())
    model.add(keras.layers.Conv2D(128, 6, strides=(2, 2)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.LeakyReLU())
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(512))
    model.add(keras.layers.LeakyReLU())
    model.add(keras.layers.Dense(6))
    model.add(keras.layers.Activation("softmax"))
    model.summary()
    return model
