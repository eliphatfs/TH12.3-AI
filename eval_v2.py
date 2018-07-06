# -*- coding: utf-8 -*-
"""
Created on Sun Jun 17 10:22:12 2018

@author: 北海若
"""

import model_v2 as mv2
import game_utils as gu
import time
import numpy as np
import dx_keycode as dxk
import tensorflow as tf


m = mv2.get_model()
oldkeystate = [False for i in range(8)]
keystate = [False for i in range(8)]  # WSADJKLs
keysetting = [dxk.DIK_W,
              dxk.DIK_S,
              dxk.DIK_A,
              dxk.DIK_D,
              dxk.DIK_J,
              dxk.DIK_K,
              dxk.DIK_L,
              dxk.DIK_SPACE]
'''keysetting = [dxk.DIK_Z,
              dxk.DIK_X,
              dxk.DIK_C,
              dxk.DIK_V,
              dxk.DIK_B,
              dxk.DIK_N,
              dxk.DIK_M,
              dxk.DIK_P]'''


def act(result, my=0):
    dis = abs(gu.fetch_posx()[my] - gu.fetch_posx()[1 - my])
    if dis < 0.24 and result // 9 == 3:
        result = 18 + (result % 9)
    if dis < 0.12 and (result // 9 == 2 or result // 9 == 3):
        result = 9 + (result % 9)
    for i in range(8):
        oldkeystate[i] = keystate[i]
    first_d = result // 9
    for i in range(4, 8):
        keystate[i] = first_d == i - 3
    next_d = (result % 9) // 3
    for i in range(2, 4):
        keystate[i] = next_d == i - 1
    last_d = result % 3
    for i in range(0, 2):
        keystate[i] = last_d == i + 1
    print(first_d, next_d, last_d)
    if ((first_d == 0 and last_d != 1)
            or (first_d == 4 and next_d == 0)):
        if gu.fetch_posx()[1 - my] - gu.fetch_posx()[my] > 0:
            oldkeystate[2] = False
            keystate[2] = True
            oldkeystate[3] = True
            keystate[3] = False
        else:
            oldkeystate[2] = True
            keystate[2] = False
            oldkeystate[3] = False
            keystate[3] = True
        if ((gu.fetch_operation()[1 - my] & 2 > 0)
                and gu.fetch_posy()[1 - my] < 0.01):
            oldkeystate[1] = False
            keystate[1] = True
            oldkeystate[0] = True
            keystate[0] = False
    # gu.write_operation(result, my)
    for i in range(8):
        if (not oldkeystate[i]) and keystate[i]:
            gu.PressKey(keysetting[i])
        if (not keystate[i]) and oldkeystate[i]:
            gu.ReleaseKey(keysetting[i])


def play(my=0):
    global Y
    en = 1 - my
    gu.update_proc()
    m.load_weights("D:/FXTZ.2.dat")
    print("Wait For Battle Detection...")
    while (gu.fetch_status() not in [0x05, 0x0d, 0x0e, 0x08, 0x09]):
        time.sleep(0.5)
    print("Battle Detected!")
    gu.update_base()
    char_act = []
    pos = []
    en_key = []
    my_key = []
    keys = [[], []]
    poses = []
    char_acts = []
    oldhp = [10000, 10000]
    while gu.fetch_hp()[0] > 0 and gu.fetch_hp()[1] > 0:
        '''if (oldhp[en] < gu.fetch_hp()[en]
                and gu.fetch_posy()[en] < 0.05
                and abs(gu.fetch_posx()[en] - 0.5) > 0.42):
            oldhp[0], oldhp[1] = gu.fetch_hp()
            gu.combo_1()
            continue'''
        oldhp[0], oldhp[1] = gu.fetch_hp()
        char_data = gu.fetch_char()
        px = gu.fetch_posx()
        py = gu.fetch_posy()
        '''if abs(px[en] - px[my]) > 0.4:
            if px[en] < px[my]:
                act(39, my)
            else:
                act(42, my)
            keys[0].append(gu.fetch_operation()[0])
            keys[1].append(gu.fetch_operation()[1])
            poses.append(np.array([px[my], py[my],
                                   px[en], py[en],
                                   px[en] - px[my],
                                   py[en] - py[my]]))
            char_acts.append(np.array([char_data[my],
                             gu.fetch_action()[my] / 100.0,
                             char_data[en],
                             gu.fetch_action()[en] / 100.0]))
            time.sleep(0.033)
            continue'''
        time_begin = time.time()
        keys[0].append(gu.fetch_operation()[my])
        keys[1].append(gu.fetch_operation()[en])
        poses.append(np.array([px[my], py[my],
                               px[en], py[en],
                               px[en] - px[my],
                               py[en] - py[my]]))
        char_acts.append(np.array([char_data[en],
                         gu.fetch_action()[en] / 100.0]))
        while len(keys[0]) > 30:
            keys[0] = keys[0][-30:]
        while len(keys[1]) > 30:
            keys[1] = keys[1][-30:]
        while len(poses) > 30:
            poses = poses[-30:]
        while len(char_acts) > 30:
            char_acts = char_acts[-30:]
        if len(keys[1]) < 30:
            continue
        char_act.append(char_acts[-1])
        pos.append(poses.copy())
        my_key.append(mv2.encode_keylist(keys[0], merge=1))
        en_key.append(mv2.encode_keylist(keys[1], merge=1))
        Y = m.predict([np.array(char_act),
                       np.array(pos)], batch_size=1)[0]
        char_act = []
        pos = []
        en_key = []
        my_key = []
        for i in range(10):
            category = np.random.choice([x for x in range(45)], p=Y[i])
            keys[0].append(gu.fetch_operation()[my])
            keys[1].append(gu.fetch_operation()[en])
            poses.append(np.array([px[my], py[my],
                                   px[en], py[en],
                                   px[en] - px[my],
                                   py[en] - py[my]]))
            char_acts.append(np.array([char_data[en],
                                       gu.fetch_action()[en] / 100.0]))
            act(category, my)
            if time_begin + 0.032 > time.time():
                time.sleep(time_begin + 0.033 - time.time())
            time_begin = time.time()
        '''category1 = np.random.choice([x for x in range(5)], p=Y[0][0])
        category2 = np.random.choice([x for x in range(3)], p=Y[1][0])
        category3 = np.random.choice([x for x in range(3)], p=Y[2][0])
        category = category1 * 9 + category2 * 3 + category3'''
        # category = np.random.choice([x for x in range(45)], p=Y)


if __name__ == "__main__":
    while 1:
        with tf.device('/cpu:0'):
            play()
