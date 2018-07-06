# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 09:27:39 2018

@author: 北海若
"""

import win32gui
import win32con
import win32process
from PIL import ImageGrab
import numpy
import time
import ctypes


# from:
# https://stackoverflow.com/questions/14489013/simulate-python-keypresses-for-controlling-a-game
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


class POperation(ctypes.Structure):
    _fields_ = [("lr", ctypes.c_long),
                ("ud", ctypes.c_long),
                ("a", ctypes.c_long),
                ("b", ctypes.c_long),
                ("c", ctypes.c_long),
                ("d", ctypes.c_long),
                ("ch", ctypes.c_long),
                ("s", ctypes.c_long)]


def PressKey(hexKeyCode):
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.ki = KeyBdInput(0, hexKeyCode, 0x0008, 0, ctypes.pointer(extra))
    x = Input(ctypes.c_ulong(1), ii_)
    ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))


def ReleaseKey(hexKeyCode):
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.ki = KeyBdInput(0, hexKeyCode,
                        0x0008 | 0x0002, 0,
                        ctypes.pointer(extra))
    x = Input(ctypes.c_ulong(1), ii_)
    ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))


kernel32 = ctypes.windll.LoadLibrary("kernel32.dll")
ReadProcessMemory = kernel32.ReadProcessMemory
WriteProcessMemory = kernel32.WriteProcessMemory
OpenProcess = kernel32.OpenProcess
_bytes = ctypes.c_ulong(0)
_root = ctypes.c_uint(0)
_baseaddr1 = ctypes.c_uint(0)
_baseaddr2 = ctypes.c_uint(0)
_bytedata = [ctypes.c_long(0) for i in range(8)]
_chardata = [ctypes.c_byte(0) for i in range(8)]
_shortdata = [ctypes.c_short(0) for i in range(8)]
_input = POperation()
hwnd = 0


def update_proc():
    global hwnd, proc
    hwnd = win32gui.FindWindow("th123_110", None)
    if hwnd:
        hreadID, processID = win32process.GetWindowThreadProcessId(hwnd)
        proc = OpenProcess(win32con.PROCESS_ALL_ACCESS, 0, processID)


def update_proc_with_pid(pid):
    global proc
    proc = OpenProcess(win32con.PROCESS_ALL_ACCESS, 0, pid)


def update_base():
    """
    Fetch base address changes caused by a new battle.
    """
    ReadProcessMemory(proc,
                      0x008855C4,  # Battle Mgr
                      ctypes.byref(_root),
                      4,
                      ctypes.byref(_bytes))
    ReadProcessMemory(proc,
                      _root.value + 0x0c,  # 1P Base
                      ctypes.byref(_baseaddr1),
                      4,
                      ctypes.byref(_bytes))
    ReadProcessMemory(proc,
                      _root.value + 0x10,  # 2P Base
                      ctypes.byref(_baseaddr2),
                      4,
                      ctypes.byref(_bytes))


def normalize_posx(raw):
    x = (raw - 1109393408) / (1151008768 - 1109393408)
    if x < 0.0410:
        return 0.0
    elif x > 0.9865:
        return 1.0
    linearized = numpy.power(numpy.e, 3.62427340 * x)
    scaled = linearized * 0.02775426
    return scaled


def normalize_posy(raw):
    if raw == 0:
        return 0
    x = (raw - 974045183) / (1143726080 - 974045183)
    if x < 0.748226:
        return 0.0
    elif x > 0.998334:
        return 1.0
    linearized = numpy.power(numpy.e, 13.387113877117 * x)
    scaled = linearized * 0.000001524590
    return scaled


def fetch_posx():
    """
    Returns
    -------------
    (1P Pos X, 2P Pos X) where values are mapped to [0, 1],
    Linear, with an absolute error of around 0.01.
    """
    ReadProcessMemory(proc,
                      _baseaddr1.value + 0xEC,
                      ctypes.byref(_bytedata[0]),
                      4,
                      ctypes.byref(_bytes))
    ReadProcessMemory(proc,
                      _baseaddr2.value + 0xEC,
                      ctypes.byref(_bytedata[1]),
                      4,
                      ctypes.byref(_bytes))
    norm_pos1 = normalize_posx(_bytedata[0].value)
    norm_pos2 = normalize_posx(_bytedata[1].value)
    return norm_pos1, norm_pos2


def fetch_posy():
    """
    Returns
    -------------
    (1P Pos Y, 2P Pos Y) where values are mapped to [0, 1],
    Linear, with an absolute error of around 0.01.
    """
    ReadProcessMemory(proc,
                      _baseaddr1.value + 0xF0,
                      ctypes.byref(_bytedata[0]),
                      4,
                      ctypes.byref(_bytes))
    ReadProcessMemory(proc,
                      _baseaddr2.value + 0xF0,
                      ctypes.byref(_bytedata[1]),
                      4,
                      ctypes.byref(_bytes))
    norm_pos1 = normalize_posy(_bytedata[0].value)
    norm_pos2 = normalize_posy(_bytedata[1].value)
    return norm_pos1, norm_pos2


def fetch_hp():
    """
    Returns
    -------------
    Integer Tuple (1P HP, 2P HP) within range [0, 10000],
    linear to HP and perfectly matches the damage in the game.
    """
    ReadProcessMemory(proc,
                      _baseaddr1.value + 0x184,
                      ctypes.byref(_bytedata[0]),
                      4,
                      ctypes.byref(_bytes))
    ReadProcessMemory(proc,
                      _baseaddr2.value + 0x184,
                      ctypes.byref(_bytedata[1]),
                      4,
                      ctypes.byref(_bytes))
    return _bytedata[0].value - 655360000, _bytedata[1].value - 655360000


def fetch_action():
    """
    Returns
    -------------
    Integer Tuple (1P Action, 2P Action).
    """
    ReadProcessMemory(proc,
                      _baseaddr1.value + 0x13C,
                      ctypes.byref(_shortdata[0]),
                      2,
                      ctypes.byref(_bytes))
    ReadProcessMemory(proc,
                      _baseaddr2.value + 0x13C,
                      ctypes.byref(_shortdata[1]),
                      2,
                      ctypes.byref(_bytes))
    return _shortdata[0].value, _shortdata[1].value


def fetch_char():
    """
    Returns
    -------------
    Integer Tuple (1P Char, 2P Char).
    """
    ReadProcessMemory(proc,
                      0x00886CF0,
                      ctypes.byref(_bytedata[0]),
                      4,
                      ctypes.byref(_bytes))
    ReadProcessMemory(proc,
                      0x00886D10,
                      ctypes.byref(_bytedata[1]),
                      4,
                      ctypes.byref(_bytes))
    return _bytedata[0].value, _bytedata[1].value


def write_operation(operation, which=0):
    first_d = operation // 9
    next_d = (operation % 9) // 3
    last_d = operation % 3

    _input.lr = 0
    _input.ud = 0
    _input.a = 0
    _input.b = 0
    _input.c = 0
    _input.d = 0
    if next_d == 1:
        _input.lr = -1
    elif next_d == 2:
        _input.lr = 1
    if last_d == 1:
        _input.ud = -1
    elif last_d == 2:
        _input.ud = 1
    if first_d == 1:
        _input.a = 1
    elif first_d == 2:
        _input.b = 1
    elif first_d == 31:
        _input.c = 1
    elif first_d == 4:
        _input.d = 1
    WriteProcessMemory(proc,
                       (_baseaddr1.value if which == 0
                        else _baseaddr2.value) + 0x754,
                       ctypes.byref(_input),
                       32,
                       ctypes.byref(_bytes))


def fetch_operation():
    """
    Returns
    -------------
    Integer Tuple (1P Operation, 2P Operation).
    The Result is in the format of Replay Data,
    But the Second byte in the Replay is the Higher byte here.
    """
    ReadProcessMemory(proc,
                      _baseaddr1.value + 0x754,
                      ctypes.byref(_input),
                      32,
                      ctypes.byref(_bytes))
    now = 0
    if (_input.ud < 0):
        now |= 1
    if (_input.ud > 0):
        now |= 2
    if (_input.lr < 0):
        now |= 4
    if (_input.lr > 0):
        now |= 8
    if (_input.a > 0):
        now |= 16
    if (_input.b > 0):
        now |= 32
    if (_input.c > 0):
        now |= 64
    if (_input.d > 0):
        now |= 128
    if (_input.ch > 0):
        now |= 256
    if (_input.s > 0):
        now |= 512
    p1 = now
    ReadProcessMemory(proc,
                      _baseaddr2.value + 0x754,
                      ctypes.byref(_input),
                      32,
                      ctypes.byref(_bytes))
    now = 0
    if (_input.ud < 0):
        now |= 1
    if (_input.ud > 0):
        now |= 2
    if (_input.lr < 0):
        now |= 4
    if (_input.lr > 0):
        now |= 8
    if (_input.a > 0):
        now |= 16
    if (_input.b > 0):
        now |= 32
    if (_input.c > 0):
        now |= 64
    if (_input.d > 0):
        now |= 128
    if (_input.ch > 0):
        now |= 256
    if (_input.s > 0):
        now |= 512
    p2 = now
    return p1, p2


def fetch_status():
    ReadProcessMemory(proc,
                      0x0088D024,
                      ctypes.byref(_bytedata[2]),
                      4,
                      ctypes.byref(_bytes))
    return _bytedata[2].value


def fetch_wincnt():
    ReadProcessMemory(proc,
                      _baseaddr1.value + 0x573,
                      ctypes.byref(_chardata[0]),
                      1,
                      ctypes.byref(_bytes))
    ReadProcessMemory(proc,
                      _baseaddr2.value + 0x573,
                      ctypes.byref(_chardata[1]),
                      1,
                      ctypes.byref(_bytes))
    return _chardata[0].value, _chardata[1].value


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
        return [0x1F]  # S
    elif action == "8":
        return [0x11]  # W
    elif action == "4":
        return [0x1E]  # A
    elif action == "6":
        return [0x20]  # D
    elif action == "3":
        return [0x1F, 0x20]
    elif action == "1":
        return [0x1F, 0x1E]
    elif action == "9":
        return [0x11, 0x20]
    elif action == "7":
        return [0x11, 0x1E]
    elif action == "A":
        return [0x24]  # J
    elif action == "B":
        return [0x25]  # K
    elif action == "C":
        return [0x26]  # L
    elif action == "D":
        return [0x39]  # Space
    elif len(action) == 2 and\
            (action[1] == "A" or
             action[1] == "B" or
             action[1] == "C" or
             action[1] == "D"):
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
    return src_image, hp_p1, hp_p2


def send_action(p):
    win32gui.ShowWindow(hwnd, win32con.SW_SHOW)
    win32gui.SetForegroundWindow(hwnd)
    time.sleep(0.02)
    if p is not None:
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

# TODO: Implement Action Set
# Normal Set:
# [L R 2 8 7 9 3 1 44 66 4D 6D 1D 2D 3D 7D 8D 9D
#  A B C 2A 2B 2C 6A 236 623 421 412]
# Minimum Set: [L R A B C Stop]


isL = False
isR = False


def act(index):  # Minimum Action Set Impl
    send_action(None)
    global isL, isR
    if index == 0:
        if not isL:
            PressKey(conv_keycode("4")[0])
            isL = True
    else:
        if isL:
            ReleaseKey(conv_keycode("4")[0])
            isL = False
    if index == 1:
        if not isR:
            PressKey(conv_keycode("6")[0])
            isR = True
    else:
        if isR:
            ReleaseKey(conv_keycode("6")[0])
            isR = False
    if index == 2:
        send_action("A")
    elif index == 3:
        send_action("B")
    elif index == 4:
        send_action("C")
    elif index == 5:
        send_action("2")
    elif index == 6:
        send_action("9")
    elif index == 7:
        send_action("7")


update_proc()
