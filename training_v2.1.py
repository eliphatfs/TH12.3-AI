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
PROCESSED_PATH = r"D:\AI_DataSet\REMIP"
BATCH = 10


def processed_data_loader(batch_size=BATCH):
    while True:
        for r, d, f in os.walk(PROCESSED_PATH):
            random.shuffle(f)
            for n in f:
                npz = np.load(os.path.join(r, n))
                d = []
                for k in range(45):
                    d.append(npz["arr_" + str(k)])
                X = [[] for i in range(4)]
                Y = []
                for l in range(500):
                    for i in range(45):
                        if len(d[i]) < 80:
                            continue
                        x, y = d[i][random.randint(0, len(d[i]) - 1)]
                        if x[0][0].shape != (30, 4):
                            continue
                        for i in range(4):
                            x[i] = x[i][0]
                            X[i].append(x[i])
                        y = y[0][0]
                        Y.append(y)
                        if len(Y) == batch_size:
                            for i in range(4):
                                X[i] = np.array(X[i])
                            Y = np.array(Y)
                            yield X, Y
                            X = [[] for i in range(4)]
                            Y = []


def data_loader(batch_size=BATCH, my_char=6, valid=False):
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
                    char_acts.append(np.array([char_data[my],
                                              int(data[my][4]) / 100.0,
                                              char_data[en],
                                              int(data[en][4]) / 100.0]))
                    if len(keys[0]) > 31:
                        keys[0] = keys[0][1:]
                    if len(keys[1]) > 31:
                        keys[1] = keys[1][1:]
                    if len(poses) > 31:
                        poses = poses[1:]
                    if len(char_acts) > 31:
                        char_acts = char_acts[1:]
                    if (len(keys[0]) == 31
                            and keys[0][-1] != keys[0][-2]
                            and keys[0][-1] & 240 == seq[st]):
                        st += 1
                        st %= len(seq)
                        char_act.append(char_acts[1:].copy())
                        pos.append(poses[-1])
                        my_key.append(mv2.encode_keylist(keys[0][:-1]))
                        en_key.append(mv2.encode_keylist(keys[1][:-1]))
                        y = mv2.key_to_category(keys[0][-1])
                        Y.append(y)
                        if len(Y) == batch_size:
                            yield ([np.array(char_act),
                                    np.array(pos),
                                    np.array(en_key),
                                    np.array(my_key)],
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
    model.fit_generator(processed_data_loader(), 1000, 40,
                        callbacks=callback,
                        validation_data=data_loader(valid=True),
                        validation_steps=50,
                        shuffle=False)


if __name__ == "__main__":
    with tf.device('/cpu:0'):
        train()
