# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 06:21:04 2018

@author: 北海若
"""
import numpy as np
import os
import model_v2 as mv2

DATA_PATH = r"D:\AI_DataSet\DATASET_REMI_TXT"


def data_loader(batch_size=1, my_char=6, valid=False):
    char_act = []
    pos = []
    en_key = []
    my_key = []
    Y = []
    for r, d, f in os.walk(DATA_PATH):
        f = sorted(f)
        if valid:
            f = f[:len(f)//10]
        else:
            f = f[len(f)//10:]
        seq = [0]
        st = 0
        tt = 0
        for n in f:
            tt += 1
            print(str(tt) + "/" + str(len(f)), "  \r", end='')
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
                        and keys[0][-1] != keys[0][-2]):
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
                               [np.array(Y)],
                               mv2.key_to_category(keys[0][-1],
                                                   one_hot=False))
                        char_act = []
                        pos = []
                        en_key = []
                        my_key = []
                        Y = []
                line = file.readline()
            file.close()


if __name__ == "__main__":
    # Train Data
    train = [[] for i in range(45)]
    sp = [0 for i in range(45)]
    dl_train = data_loader()
    part = 1
    while True:
        try:
            dx, dy, k = next(dl_train)
            train[k].append((dx, dy))
            sp[k] += 1
            if sum([len(train[i]) for i in range(45)]) > 60000:
                np.savez_compressed("D:/AI_DataSet/REMIP/processed_data_%d.npz" % part, *train)
                train = [[] for i in range(45)]
                part += 1
        except StopIteration:
            break
    f = open("D:/processed_data.txt", "w")
    f.write(str(sp))
    f.close()
