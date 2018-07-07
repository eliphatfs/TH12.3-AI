# -*- coding: utf-8 -*-
"""
Created on Fri Jul  6 15:30:32 2018

@author: 北海若
"""

import keras
from keras import layers
import numpy as np
import rl.core
import rl.memory
import rl.policy
import rl.agents.dqn
import rl.callbacks
import game_utils as gu
import dx_keycode as dxk
import time


oldkeystate = [False for i in range(8)]
keystate = [False for i in range(8)]  # WSADJKLs
keysetting = ([dxk.DIK_W,
               dxk.DIK_S,
               dxk.DIK_A,
               dxk.DIK_D,
               dxk.DIK_J,
               dxk.DIK_K,
               dxk.DIK_L,
               dxk.DIK_SPACE],
              [dxk.DIK_T,
               dxk.DIK_Y,
               dxk.DIK_C,
               dxk.DIK_V,
               dxk.DIK_B,
               dxk.DIK_N,
               dxk.DIK_M,
               dxk.DIK_P])


def act(result, my=0):
    dis = gu.fetch_posx()[my] - gu.fetch_posx()[1 - my]
    if dis > 0.45:
        result = 39
    elif dis < -0.45:
        result = 42
    for i in range(8):
        oldkeystate[i] = keystate[i]
    print("", result, my)
    first_d = result // 9
    for i in range(4, 8):
        keystate[i] = first_d == i - 3
    next_d = (result % 9) // 3
    for i in range(2, 4):
        keystate[i] = next_d == i - 1
    last_d = result % 3
    for i in range(0, 2):
        keystate[i] = last_d == i + 1
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
            gu.PressKey(keysetting[my][i])
        if (not keystate[i]) and oldkeystate[i]:
            gu.ReleaseKey(keysetting[my][i])


def key_to_category(key, one_hot=True, new=False):
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
        return np.eye(45)[cjkld * 9 + cad * 3 + cws]
    else:
        return cjkld * 9 + cad * 3 + cws


def encode_keylist(list_key, merge=1, one_hot=True, new=False):
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
    a = layers.Dense(128, activation='softmax')(a)
    a_probs = layers.Permute((2, 1))(a)
    output_attention_mul = layers.Multiply()([inputs, a_probs])
    return output_attention_mul


def conv1d_block(*args, **kwargs):
    def conv1d_get_tensor(inputs):
        x = layers.Conv1D(*args, **kwargs)(inputs)
        x = layers.LeakyReLU()(x)
        x = layers.BatchNormalization()(x)
        return x
    return conv1d_get_tensor


def wavenet_block(n_atrous_filters, atrous_filter_size, atrous_rate):
    def f(input_):
        residual = input_
        tanh_out = layers.Conv1D(n_atrous_filters, atrous_filter_size,
                                 dilation_rate=atrous_rate,
                                 padding='causal',
                                 activation='tanh')(input_)
        sigmoid_out = layers.Conv1D(n_atrous_filters, atrous_filter_size,
                                    dilation_rate=atrous_rate,
                                    padding='causal',
                                    activation='sigmoid')(input_)
        merged = layers.Multiply()([tanh_out, sigmoid_out])
        merged = layers.BatchNormalization()(merged)
        skip_out = layers.Conv1D(24, 1)(merged)
        skip_out = layers.LeakyReLU()(skip_out)
        skip_out = layers.BatchNormalization()(skip_out)
        out = layers.Add()([skip_out, residual])
        return out, skip_out
    return f


def get_model():
    inp = layers.Input(shape=[1, 128, 100])
    inp_a = layers.Reshape([128, 100])(inp)
    inp_a = attention_3d_block(inp_a)
    '''char_action = layers.Input(shape=[128, 4])
    char_action_a = attention_3d_block(char_action)

    position = layers.Input(shape=[128, 6])
    position_a = attention_3d_block(position)
    # position_r = layers.RepeatVector(30)(position)

    enemy_key = layers.Input(shape=[128, 45])
    enemy_key_a = attention_3d_block(enemy_key)

    my_key = layers.Input(shape=[128, 45])
    my_key_a = attention_3d_block(my_key)
    concat = layers.Concatenate()([char_action_a, position_a,
                                   enemy_key_a, my_key_a])'''
    # gate = layers.Dense(100, activation="sigmoid")(concat)
    # concat = layers.Multiply()([gate, concat])
    first = conv1d_block(24, 4, padding='causal')(inp_a)
    A, B = wavenet_block(32, 2, 1)(first)
    skip_connections = [B]
    for i in range(1, 16):
        A, B = wavenet_block(32, 2, 2 ** (i % 4))(A)
        skip_connections.append(B)
    net = layers.Add()(skip_connections)
    net = layers.LeakyReLU()(net)
    net = conv1d_block(4, 1)(net)
    net = layers.LeakyReLU()(net)
    net = layers.Flatten()(net)
    net = layers.Dense(45, activation='linear')(net)
    return keras.models.Model(inputs=inp,
                              outputs=net)


def get_model_against():
    global m1, m2
    char_action_p1 = layers.Input(shape=[128, 4])
    char_action_p2 = layers.Input(shape=[128, 4])
    position_p1 = layers.Input(shape=[128, 6])
    position_p2 = layers.Input(shape=[128, 6])
    p1_key = layers.Input(shape=[128, 45])
    p2_key = layers.Input(shape=[128, 45])
    m1 = get_model()
    m2 = get_model()
    o1 = m1(char_action_p1, position_p1, p2_key, p1_key)
    o2 = m2(char_action_p2, position_p2, p1_key, p2_key)
    o1 = layers.Reshape([45, 1])(o1)
    o2 = layers.Reshape([1, 45])(o2)
    o = keras.backend.batch_dot(o1, o2)
    return keras.models.Model(inputs=[char_action_p1,
                                      char_action_p2,
                                      position_p1,
                                      position_p2,
                                      p1_key,
                                      p2_key],
                              outputs=o)


class TH123Env(rl.core.Env):

    def __init__(self):
        self.p1_char_acts = []
        self.p1_positions = []
        self.p2_char_acts = []
        self.p2_positions = []
        self.p1_keys = []
        self.p2_keys = []
        self.old_dhp = 0  # P1 - P2
        self.current_act = 0

    def step(self, action):
        gu.press_key([dxk.DIK_Z])
        gu.update_base()
        act(action, self.current_act)
        while len(self.p1_keys) <= 128:
            time.sleep(0.05)
            char_data = gu.fetch_char()
            px = gu.fetch_posx()
            py = gu.fetch_posy()
            self.p1_keys.append(gu.fetch_operation()[0])
            self.p2_keys.append(gu.fetch_operation()[1])
            self.p1_positions.append(np.array([px[0], py[0],
                                               px[1], py[1],
                                               px[1] - px[0],
                                               py[1] - py[0]]))
            self.p2_positions.append(np.array([px[1], py[1],
                                               px[0], py[0],
                                               px[0] - px[1],
                                               py[0] - py[1]]))
            self.p1_char_acts.append(np.array([char_data[0],
                                               gu.fetch_action()[0],
                                               char_data[1],
                                               gu.fetch_action()[1]]))
            self.p2_char_acts.append(np.array([char_data[1],
                                               gu.fetch_action()[1],
                                               char_data[0],
                                               gu.fetch_action()[0]]))
        while len(self.p1_keys) > 128:
            self.p1_keys = self.p1_keys[-128:]
        while len(self.p2_keys) > 128:
            self.p2_keys = self.p2_keys[-128:]
        while len(self.p1_positions) > 128:
            self.p1_positions = self.p1_positions[-128:]
        while len(self.p2_positions) > 128:
            self.p2_positions = self.p2_positions[-128:]
        while len(self.p1_char_acts) > 128:
            self.p1_char_acts = self.p1_char_acts[-128:]
        while len(self.p2_char_acts) > 128:
            self.p2_char_acts = self.p2_char_acts[-128:]
        reward = 0
        maybe_rwd = -(self.old_dhp - (gu.fetch_hp()[0] - gu.fetch_hp()[1]))
        if self.old_dhp - (gu.fetch_hp()[0] - gu.fetch_hp()[1]) > 100:
            reward = maybe_rwd / 200.0
        elif self.old_dhp - (gu.fetch_hp()[0] - gu.fetch_hp()[1]) < -100:
            reward = maybe_rwd / 200.0
        if gu.fetch_hp()[0] > gu.fetch_hp()[1]:
            reward = 0.01
        elif gu.fetch_hp()[0] < gu.fetch_hp()[1]:
            reward = -0.01
        if gu.fetch_hp()[0] == 0 and gu.fetch_hp()[1] > 0:
            reward = -5000.0
        elif gu.fetch_hp()[0] > 0 and gu.fetch_hp()[1] == 0:
            reward = 5000.0
        if self.current_act == 1:
            reward = -reward
        self.old_dhp = (gu.fetch_hp()[0] - gu.fetch_hp()[1])
        state = None
        if self.current_act == 1:
            self.current_act = 0
            state = np.concatenate((np.array(self.p2_char_acts),
                                    np.array(self.p2_positions),
                                    encode_keylist(self.p1_keys),
                                    encode_keylist(self.p2_keys)), axis=-1)
        elif self.current_act == 0:
            self.current_act = 1
            state = np.concatenate((np.array(self.p1_char_acts),
                                    np.array(self.p1_positions),
                                    encode_keylist(self.p2_keys),
                                    encode_keylist(self.p1_keys)), axis=-1)
        # gu.press_key([dxk.DIK_ESCAPE])
        return state, reward, abs(reward) > 999, {}

    def reset(self):
        gu.update_proc()
        gu.send_action("A")
        gu.press_key([dxk.DIK_ESCAPE])
        time.sleep(0.1)
        gu.press_key([dxk.DIK_UP])
        time.sleep(0.1)
        while (gu.fetch_status() not in [0x03]):
            gu.press_key([dxk.DIK_Z])
            time.sleep(0.5)
        for i in range(10):
            gu.press_key([dxk.DIK_A])
            time.sleep(0.2)
        while (gu.fetch_status() not in [0x05, 0x0d, 0x0e, 0x08, 0x09]):
            gu.press_key([dxk.DIK_Z])
            time.sleep(0.2)
        self.p1_char_acts = []
        self.p1_positions = []
        self.p2_char_acts = []
        self.p2_positions = []
        self.p1_keys = []
        self.p2_keys = []
        self.old_dhp = 0
        gu.update_base()
        return self.step(0)[0]

    def render(self, mode='human', close=False):
        pass

    def close(self):
        pass


class ModelCheckpoint(rl.callbacks.Callback):

    def __init__(self):
        self.total_steps = 0

    def on_step_end(self, step, logs={}):
        self.total_steps += 1
        if self.total_steps % 3600 != 0:
            return

        self.model.save_weights('D:/FXTZ.DQN.dat', overwrite=True)


def train():
    global env
    env = TH123Env()
    memory = rl.memory.SequentialMemory(18000, window_length=1)
    policy = rl.policy.LinearAnnealedPolicy(rl.policy.EpsGreedyQPolicy(),
                                            'eps',
                                            0.11,
                                            0.1,
                                            0.05,
                                            800000)
    m = get_model()
    m.summary()
    m.load_weights("D:/FXTZ.DQN.dat")
    dqn = rl.agents.dqn.DQNAgent(model=m,
                                 batch_size=32,
                                 nb_actions=45,
                                 policy=policy,
                                 memory=memory,
                                 nb_steps_warmup=7200,
                                 gamma=0.99,
                                 target_model_update=50000,
                                 train_interval=8)
    dqn.compile(keras.optimizers.Adadelta(), metrics=['mae'])
    callbacks = [ModelCheckpoint()]
    callbacks += [rl.callbacks.FileLogger("dql_training.log", interval=1)]
    dqn.fit(env,
            callbacks=callbacks,
            nb_steps=20000000,
            log_interval=900)


if __name__ == "__main__":
    train()
