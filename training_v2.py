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


DATA_PATH = r"D:\AI_DataSet\DATASET_REMI_TXT"


def data_loader(batch_size=32, my_char=6):
    char_act = []
    pos = []
    en_key = []
    my_key = []
    Y1 = []
    Y2 = []
    Y3 = []
    while True:
        for r, d, f in os.walk(DATA_PATH):
            random.shuffle(f)
            seq = [0] * 2 + [16] * 9 + [32, 64, 128, 1] * 2
            st = 0
            for n in f:
                st += 1
                st %= len(seq)
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
                line = file.readline()
                while line != "":
                    data = line.split("; ")
                    data[0] = data[0].split(" ")
                    data[1] = data[1].split(" ")
                    if int(data[0][0]) < 0 or int(data[1][0]) < 0:
                        break
                    keys[0].append(int(data[my][3]))
                    keys[1].append(int(data[en][3]))
                    if len(keys[0]) > 60:
                        keys[0] = keys[0][1:]
                    if len(keys[1]) > 60:
                        keys[1] = keys[1][1:]
                    if (len(keys[0]) == 60
                            and keys[0][-1] != keys[0][-2]
                            and keys[0][-1] & seq[st] == seq[st]):
                        char_act.append(np.array([char_data[my],
                                                  int(data[my][4]),
                                                  char_data[en],
                                                  int(data[en][4])]))
                        pxmy = float(data[my][1])
                        pymy = float(data[my][2])
                        pxen = float(data[en][1])
                        pyen = float(data[en][2])
                        pos.append(np.array([pxmy, pymy,
                                             pxen, pyen,
                                             pxen - pxmy,
                                             pyen - pymy]))
                        my_key.append(mv2.encode_keylist(keys[0][:-1]))
                        en_key.append(mv2.encode_keylist(keys[1][:-1]))
                        y1, y2, y3 = mv2.key_to_category(keys[0][-1], new=True)
                        Y1.append(y1)
                        Y2.append(y2)
                        Y3.append(y3)
                        if len(Y1) == batch_size:
                            yield ([np.array(char_act),
                                    np.array(pos),
                                    np.array(en_key),
                                    np.array(my_key)],
                                   [np.array(Y1),
                                    np.array(Y2),
                                    np.array(Y3)])
                            char_act = []
                            pos = []
                            en_key = []
                            my_key = []
                            Y1 = []
                            Y2 = []
                            Y3 = []
                    line = file.readline()
                file.close()


def train():
    global model
    model = mv2.get_model()
    model.compile("adadelta",
                  "categorical_crossentropy",
                  ["acc"])
    model.summary()
    '''try:
        model.load_weights("D:/FXTZ.dat")
    except Exception:
        pass'''
    callback = [callbacks.CSVLogger("training.csv"),
                callbacks.ModelCheckpoint("D:/FXTZ.dat",
                                          save_weights_only=True)]
    model.fit_generator(data_loader(), 300, 40,
                        callbacks=callback,
                        validation_data=data_loader(),
                        validation_steps=25,
                        shuffle=False)


if __name__ == "__main__":
    train()
