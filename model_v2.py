# -*- coding: utf-8 -*-
"""
Created on Sun Jun 17 07:54:53 2018

@author: 北海若
"""

import keras
from keras import layers
import numpy as np


N_CLASS = 45


def key_to_category(key, one_hot=False, new=False):
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
    if new:
        return np.eye(5)[cjkld], np.eye(3)[cad], np.eye(3)[cws]
    elif one_hot:
        return np.eye(N_CLASS)[cjkld * 9 + cad * 3 + cws]
    else:
        return cjkld * 9 + cad * 3 + cws


def encode_keylist(list_key, merge=2, one_hot=True, new=False):
    list_key = list_key.copy()
    tmp = []
    for i in range(len(list_key)):
        list_key[i] = key_to_category(list_key[i], one_hot=one_hot, new=new)
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
    return np.array(list_key)


def attention_3d_block(inputs):
    a = layers.Permute((2, 1))(inputs)
    a = layers.Dense(30, activation='softmax')(a)
    a_probs = layers.Permute((2, 1))(a)
    output_attention_mul = layers.Multiply()([inputs, a_probs])
    return output_attention_mul


def conv1d_block(*args, **kwargs):
    def conv1d_get_tensor(inputs):
        x = layers.Conv1D(*args, **kwargs)(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        return x
    return conv1d_get_tensor


def get_model():
    char_action = layers.Input(shape=[4])
    repeated_action = layers.RepeatVector(45)(char_action)

    position = layers.Input(shape=[45, 6])

    enemy_key = layers.Input(shape=[45, 45])

    my_key = layers.Input(shape=[45, 45])
    concat = layers.Concatenate()([repeated_action, position,
                                   enemy_key, my_key])
    gate = layers.Dense(100, activation="sigmoid")(concat)
    concat = layers.Multiply()([gate, concat])
    c1 = conv1d_block(128, 1)(concat)
    c2 = conv1d_block(128, 3, padding="causal")(c1)
    c3 = conv1d_block(128, 3, padding="causal", dilation_rate=2)(c2)
    c4 = conv1d_block(128, 3, padding="causal", dilation_rate=4)(c3)
    c5 = conv1d_block(128, 3, padding="causal", dilation_rate=8)(c4)
    c5 = conv1d_block(128, 3, padding="causal", dilation_rate=16)(c4)
    c6 = conv1d_block(8, 1)(c5)
    c6 = layers.Flatten()(c6)
    dense = layers.Dense(256, activation='relu')(c6)
    dense_category_1 = layers.Dense(5, activation='softmax')(dense)
    dense_category_2 = layers.Dense(3, activation='softmax')(dense)
    dense_category_3 = layers.Dense(3, activation='softmax')(dense)
    return keras.models.Model(inputs=[char_action,
                                      position,
                                      enemy_key,
                                      my_key],
                              outputs=[dense_category_1,
                                       dense_category_2,
                                       dense_category_3],
                              name="TH123AI")
