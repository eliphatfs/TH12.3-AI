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


def encode_keylist(list_key, merge=3, one_hot=True):
    list_key = list_key.copy()
    tmp = []
    for i in range(len(list_key)):
        list_key[i] = key_to_category(list_key[i])
    for i in range(merge):
        for j in range(len(list_key)):
            if j > 0 and list_key[j] == list_key[j - 1]:
                continue
            tmp.append(list_key[j])
        list_key = tmp
        tmp = []
    if one_hot:
        for i in range(len(list_key)):
            list_key[i] = np.eye(N_CLASS)[list_key[i]]
    return np.array(list_key)


def get_model():
    char_action = layers.Input(shape=[4])
    # [Char1, Action1, Char2, Action2]
    char_action_dnn = layers.Dense(128)(char_action)
    char_action_dnn = layers.LeakyReLU()(char_action_dnn)
    for _ in range(1):
        char_action_dnn = layers.Dense(128)(char_action_dnn)
        char_action_dnn = layers.LeakyReLU()(char_action_dnn)

    position = layers.Input(shape=[6])
    # [MyPosX, MyPosY, EnPosX, EnPosY, DeltaPosX, DeltaPosY]
    position_dnn = layers.Dense(128)(position)
    position_dnn = layers.LeakyReLU()(position_dnn)
    for _ in range(1):
        position_dnn = layers.Dense(128)(position_dnn)
        position_dnn = layers.LeakyReLU()(position_dnn)

    enemy_key = layers.Input(shape=[None, 45])
    enemy_key_rnn = layers.LSTM(128, recurrent_dropout=0.3)(enemy_key)

    my_key = layers.Input(shape=[None, 45])
    my_key_rnn = layers.LSTM(128, recurrent_dropout=0.3)(my_key)

    final_merge = layers.Add()([char_action_dnn, position_dnn,
                                enemy_key_rnn, my_key_rnn])
    final_dnn = layers.Dense(128)(final_merge)
    final_dnn = layers.LeakyReLU()(final_dnn)
    final_dnn = layers.Dense(45)(final_dnn)
    final_dnn = layers.Activation("softmax")(final_dnn)
    return keras.models.Model(inputs=[char_action,
                                      position,
                                      enemy_key,
                                      my_key],
                              outputs=[final_dnn],
                              name="TH123AI")
