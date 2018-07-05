# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 20:46:28 2018

@author: 北海若
"""

import sklearn.ensemble as sk
import sklearn.metrics as metrics
import os
import random
import numpy as np
import time
import pickle

DATA_PATH = r"D:\AI_DataSet\DATASET_REMI_TXT"
TRAIN = 0
EVAL = 1
MODE = EVAL


def get_model():
    estim = sk.RandomForestClassifier(n_estimators=100, max_depth=10)
    return estim


m = get_model()


def data_loader(batch_size=8, my_char=6, valid=False):
    import model_v2 as mv2
    X = []
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
                    poses = np.array([pxmy, pymy,
                                      pxen, pyen,
                                      pxen - pxmy,
                                      pyen - pymy])
                    char_acts = np.array([char_data[my],
                                         int(data[my][4]) / 100.0,
                                         char_data[en],
                                         int(data[en][4]) / 100.0])
                    if len(keys[0]) > 31:
                        keys[0] = keys[0][1:]
                    if len(keys[1]) > 31:
                        keys[1] = keys[1][1:]
                    if (len(keys[0]) == 31
                            and keys[0][-1] != keys[0][-2]
                            and keys[0][-1] & 240 == seq[st]):
                        st += 1
                        st %= len(seq)
                        X.append(np.concatenate((char_acts, poses)))
                        y = mv2.key_to_category(keys[0][-1], one_hot=False)
                        Y.append(y)
                        if len(Y) % 1000 == 0:
                            print("Preparing: %d / %d" % (len(Y), batch_size))
                        if len(Y) == batch_size:
                            yield (np.array(X),
                                   np.array(Y))
                            X = []
                            Y = []
                    line = file.readline()
                file.close()


def train():
    dl = data_loader(batch_size=200000)

    X, Y = next(dl)
    m.fit(X, Y)
    f = open("D:/FXTZ.RT/FXTZ.RT.dat", "wb")
    pickle.dump(m, f)
    f.close()
    print(metrics.make_scorer(metrics.accuracy_score)(m, X, Y))


def evaluate(my=0):
    global m
    import game_utils as gu
    from eval_v2 import act
    en = 1 - my
    f = open("D:/FXTZ.RT/FXTZ.RT.dat", "rb")
    m = pickle.load(f)
    f.close()
    print("Wait For Battle Detection...")
    while (gu.fetch_status() not in [0x05, 0x0d, 0x0e, 0x08, 0x09]):
        time.sleep(0.5)
    print("Battle Detected!")
    gu.update_base()
    while gu.fetch_hp()[0] > 0 and gu.fetch_hp()[1] > 0:
        char_data = gu.fetch_char()
        px = gu.fetch_posx()
        py = gu.fetch_posy()

        inputs = np.array([[char_data[my],
                            gu.fetch_action()[my] / 100.0,
                            char_data[en],
                            gu.fetch_action()[en] / 100.0,
                            px[my], py[my],
                            px[en], py[en],
                            px[en] - px[my],
                            py[en] - py[my]]])
        category = m.predict(inputs)[0]
        act(category, my)
        time.sleep(0.033)


if __name__ == "__main__":
    if MODE == TRAIN:
        train()
    if MODE == EVAL:
        evaluate()
