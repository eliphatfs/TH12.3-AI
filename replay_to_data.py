# -*- coding: utf-8 -*-
"""
Created on Sun Jun 10 12:03:42 2018

@author: 北海若
"""

import game_utils as gu
import time


def replay_to_data():
    gu.update_base()
    hp1, hp2 = 10000, 10000
    data = []
    f = open(str(int(time.time() * 1000)) + ".txt", "w+")
    f.write("P1: ")
    f.write(str(gu.fetch_char()[0]))
    f.write(", ")
    f.write("P2: ")
    f.write(str(gu.fetch_char()[1]))
    f.write("\n")
    while hp1 > 0 and hp2 > 0:
        hp1, hp2 = gu.fetch_hp()
        key1, key2 = gu.fetch_operation()
        pos1x, pos2x = gu.fetch_posx()
        pos1y, pos2y = gu.fetch_posy()
        data.append(hp1)
        data.append(round(pos1x, 4))
        data.append(round(pos1y, 4))
        data.append(key1)
        data.append(hp2)
        data.append(round(pos2x, 4))
        data.append(round(pos2y, 4))
        data.append(key2)
        time.sleep(1.0 / 20.0)
    if hp1 <= 0 and hp2 > 0:
        f.write("P2 Won.")
    elif hp2 <= 0 and hp1 > 0:
        f.write("P1 Won.")
    elif hp2 <= 0 and hp1 <= 0:
        f.write("Tie.")
    else:
        f.write("Error Occurred, Player Won Unknown.")
    f.write("\n# P1 HP P1 Pos X P1 Pos Y P1 Key; P2 etc.\n")
    for x in range(0, len(data), 8):
        for i in range(x, x + 4):
            f.write(str(data[i]))
            f.write(" ")
        f.write(";")
        for i in range(x + 4, x + 8):
            f.write(str(data[i]))
            f.write(" ")
        f.write("\n")
    f.close()


if __name__ == "__main__":
    replay_to_data()
