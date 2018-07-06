# -*- coding: utf-8 -*-
"""
Created on Sun Jun 17 07:54:24 2018

@author: 北海若
"""

import model_v2 as mv2
import os
import numpy as np
import random
from keras import callbacks
import tensorflow as tf


DATA_PATH = r"D:\AI_DataSet\DATASET_REMI_TXT"


def data_loader(batch_size=8, my_char=7, valid=False):
    char_act = []
    pos = []
    en_key = []
    my_key = []
    Y = []
    while True:
        for r, d, f in os.walk(DATA_PATH):
            f = sorted(f)
            if valid:
                f = f[:len(f)//10]
            else:
                f = f[len(f)//10:]
            random.shuffle(f)
            seq = [0, 16, 32, 64, 128]
            st = 0
            for n in f:
                file = open(os.path.join(r, n), "rt")
                char_data = file.readline()
                win_lose = file.readline()
                char_data = char_data.split(", ")
                char_data[0] = int(char_data[0].split(": ")[-1])
                char_data[1] = int(char_data[1].split(": ")[-1])
                my = -1
                if char_data[0] == my_char and win_lose.startswith("P1"):
                    my = 0
                if char_data[1] == my_char and win_lose.startswith("P2"):
                    my = 1
                if my == -1:
                    continue
                en = 1 - my
                file.readline()
                keys = [[], []]
                poses = []
                char_acts = []
                line = file.readline()
                while line != "":
                    data = line.split("; ")
                    data[0] = data[0].split(" ")
                    data[1] = data[1].split(" ")
                    if int(data[0][0]) < 0 or int(data[1][0]) < 0:
                        break
                    keys[0].append(int(data[my][3]))
                    keys[1].append(int(data[en][3]))
                    pxmy = float(data[my][1])
                    pymy = float(data[my][2])
                    pxen = float(data[en][1])
                    pyen = float(data[en][2])
                    poses.append(np.array([pxmy, pymy,
                                           pxen, pyen,
                                           pxen - pxmy,
                                           pyen - pymy]))
                    char_acts.append(np.array([char_data[en],
                                              int(data[en][4]) / 100.0]))
                    if len(keys[0]) > 60:
                        keys[0] = keys[0][1:]
                    if len(keys[1]) > 60:
                        keys[1] = keys[1][1:]
                    if len(poses) > 60:
                        poses = poses[1:]
                    if len(char_acts) > 60:
                        char_acts = char_acts[1:]
                    if (len(keys[0]) == 60
                            and keys[0][-1] != keys[0][-2]
                            and keys[0][30] & 240 == seq[st]):
                        st += 1
                        st %= len(seq)
                        char_act.append(char_acts[29])
                        pos.append(poses[:30])
                        my_key.append(mv2.encode_keylist(keys[0][:-1]))
                        en_key.append(mv2.encode_keylist(keys[1][:-1]))
                        y = mv2.encode_keylist(keys[0][30:])
                        Y.append(y)
                        if len(Y) == batch_size:
                            yield ([np.array(char_act),
                                    np.array(pos)],
                                   [np.array(Y)])
                            char_act = []
                            pos = []
                            en_key = []
                            my_key = []
                            Y = []
                    line = file.readline()
                file.close()


def train():
    global model
    model = mv2.get_model()
    model.compile("adam",
                  "categorical_crossentropy",
                  ["acc"])
    model.summary()
    '''try:
        model.load_weights("D:/FXTZ.dat")
    except Exception:
        pass'''
    callback = [callbacks.CSVLogger("training.csv"),
                callbacks.ModelCheckpoint("D:/FXTZ.dat",
                                          save_weights_only=True,
                                          save_best_only=True)]
    model.fit_generator(data_loader(), 300, 40,
                        callbacks=callback,
                        validation_data=data_loader(valid=True),
                        validation_steps=25,
                        shuffle=False)


if __name__ == "__main__":
    with tf.device('/cpu:0'):
        train()
