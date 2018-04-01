# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 09:27:39 2018

@author: 北海若
"""

import win32gui
import win32con
from PIL import ImageGrab
import numpy
import time
import ctypes


# from: https://stackoverflow.com/questions/14489013/simulate-python-keypresses-for-controlling-a-game
SendInput = ctypes.windll.user32.SendInput

# C struct redefinitions
PUL = ctypes.POINTER(ctypes.c_ulong)


class KeyBdInput(ctypes.Structure):
    _fields_ = [("wVk", ctypes.c_ushort),
                ("wScan", ctypes.c_ushort),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", PUL)]


class HardwareInput(ctypes.Structure):
    _fields_ = [("uMsg", ctypes.c_ulong),
                ("wParamL", ctypes.c_short),
                ("wParamH", ctypes.c_ushort)]


class MouseInput(ctypes.Structure):
    _fields_ = [("dx", ctypes.c_long),
                ("dy", ctypes.c_long),
                ("mouseData", ctypes.c_ulong),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", PUL)]


class Input_I(ctypes.Union):
    _fields_ = [("ki", KeyBdInput),
                ("mi", MouseInput),
                ("hi", HardwareInput)]


class Input(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong),
                ("ii", Input_I)]


def PressKey(hexKeyCode):
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.ki = KeyBdInput(0, hexKeyCode, 0x0008, 0, ctypes.pointer(extra))
    x = Input(ctypes.c_ulong(1), ii_)
    ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))


def ReleaseKey(hexKeyCode):
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.ki = KeyBdInput(0, hexKeyCode, 0x0008 | 0x0002, 0, ctypes.pointer(extra))
    x = Input(ctypes.c_ulong(1), ii_)
    ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))


hwnd = win32gui.FindWindow("th123_110", "东方非想天则 ～ 追寻特大型人偶之谜 Ver1.10(beta)")
if not hwnd:
    print('window not found!')
else:
    print(hwnd)


def press_key(code):
    for c in code:
        if type(c) == list:
            for cc in c:
                PressKey(cc)
            time.sleep(0.03)
            for cc in c:
                ReleaseKey(cc)
        else:
            PressKey(c)
            time.sleep(0.03)
            ReleaseKey(c)
        time.sleep(0.03)
        print(c)


def conv_keycode(action):
    if action == "2":
        return [0x1F]
    elif action == "8":
        return [0x11]
    elif action == "4":
        return [0x1E]
    elif action == "6":
        return [0x20]
    elif action == "3":
        return [0x1F, 0x20]
    elif action == "1":
        return [0x1F, 0x1E]
    elif action == "9":
        return [0x11, 0x20]
    elif action == "7":
        return [0x11, 0x1E]
    elif action == "A":
        return [0x24]
    elif action == "B":
        return [0x25]
    elif action == "C":
        return [0x26]
    elif action == "D":
        return [0x39]
    elif len(action) == 2 and (action[1] == "A" or action[1] == "B" or action[1] == "C" or action[1] == "D"):
        o_list = []
        for i in conv_keycode(action[0]):
            o_list.append(i)
        o_list.append(conv_keycode(action[1])[0])
        return [o_list]
    last_order_list = []
    for x in action:
        last_order_list.append(conv_keycode(x))
    return last_order_list


def fetch_screen():
    game_rect = win32gui.GetWindowRect(hwnd)
    src_image = ImageGrab.grab(game_rect)
    src_array = numpy.asarray(src_image)
    hp_p1 = 0
    hp_p2 = 0
    for i in range(280):
        if int(src_array[75, i, 0]) + int(src_array[75, i, 1]) > 300:
            hp_p1 += 1
    for i in range(369, 639):
        if int(src_array[75, i, 0]) + int(src_array[75, i, 1]) > 300:
            hp_p2 += 1
    print(hp_p1, hp_p2)
    return src_image


def send_action(p):
    win32gui.ShowWindow(hwnd, win32con.SW_SHOW)
    win32gui.SetForegroundWindow(hwnd)
    time.sleep(0.02)
    press_key(conv_keycode(p))


def combo_1():
    send_action("A" * 15)
    PressKey(conv_keycode("2")[0])
    send_action("A" * 18)
    send_action("C" * 3)
    time.sleep(0.2)
    ReleaseKey(conv_keycode("2")[0])


def combo_2():
    send_action("2")
    PressKey(conv_keycode("2")[0])
    time.sleep(0.2)
    send_action("A" * 2)
    ReleaseKey(conv_keycode("2")[0])
    for i in range(10):
        send_action("9C")


def combo_3():
    send_action("C")
    time.sleep(0.3)
    send_action("29")
    time.sleep(0.2)
    PressKey(conv_keycode("2")[0])
    time.sleep(0.1)
    send_action("C")
    ReleaseKey(conv_keycode("2")[0])
    time.sleep(0.3)
    send_action("66")
    send_action("A" * 14)
    PressKey(conv_keycode("2")[0])
    # time.sleep(0.5)
    send_action("A" * 8)
    ReleaseKey(conv_keycode("2")[0])
