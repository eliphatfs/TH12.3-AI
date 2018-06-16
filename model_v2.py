# -*- coding: utf-8 -*-
"""
Created on Sun Jun 17 07:54:53 2018

@author: 北海若
"""

import keras
from keras import layers
import numpy as np


N_CLASS = 45


def key_to_category(key, one_hot=False):
    cws = 0
    cad = 0
    cjkld = 0
    if key & 1 > 0:  # W
        cws = 1
    if key & 2 > 0:
        cws = 2
    if key & 4 > 0:
        cad = 1
    if key & 8 > 0:
        cad = 2
    if key & 64 > 0:
        cjkld = 3
    if key & 128 > 0:
        cjkld = 4
    if key & 16 > 0:
        cjkld = 1
    if key & 32 > 0:
        cjkld = 2
    if one_hot:
        return np.eye(N_CLASS)[cjkld * 9 + cad * 3 + cws]
    else:
        return cjkld * 9 + cad * 3 + cws


def encode_keylist(list_key, merge=2, one_hot=True):
    list_key = list_key.copy()
    tmp = []
    for i in range(len(list_key)):
        list_key[i] = key_to_category(list_key[i])
    for i in range(0, len(list_key), merge):
        tmp.append(list_key[i])
    list_key = tmp
    '''for i in range(merge):
        for j in range(len(list_key)):
            if j > 0 and list_key[j] == list_key[j - 1]:
                continue
            tmp.append(list_key[j])
        list_key = tmp
        tmp = []'''
    if one_hot:
        for i in range(len(list_key)):
            list_key[i] = np.eye(N_CLASS)[list_key[i]]
    return np.array(list_key)


def get_model():
    char_action = layers.Input(shape=[4])
    repeated_action = layers.RepeatVector(30)(char_action)
    dnn_action = layers.TimeDistributed(layers.Dense(45))(repeated_action)
    dnn_action = layers.BatchNormalization()(dnn_action)
    dnn_action = layers.Activation("tanh")(dnn_action)

    position = layers.Input(shape=[6])
    repeated_position = layers.RepeatVector(30)(position)
    dnn_pos = layers.TimeDistributed(layers.Dense(45))(repeated_position)
    dnn_pos = layers.Activation("tanh")(dnn_pos)

    enemy_key = layers.Input(shape=[30, 45])
    my_key = layers.Input(shape=[30, 45])
    concat = layers.Concatenate()([dnn_action, dnn_pos,
                                   enemy_key, my_key])
    flatten = layers.Flatten()(concat)
    dense = layers.Dense(256, activation="tanh")(flatten)
    dense = layers.Dense(256, activation="tanh")(dense)
    dense = layers.Dense(128, activation="tanh")(dense)
    dense = layers.Dense(128, activation="tanh")(dense)
    dense_category = layers.Dense(45, activation='softmax')(dense)
    return keras.models.Model(inputs=[char_action,
                                      position,
                                      enemy_key,
                                      my_key],
                              outputs=[dense_category],
                              name="TH123AI")
