# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 11:53:16 2018

@author: 北海若
"""

import game_utils
import model
import numpy
import keras
import copy


m = model.get_model()
memory = []
gamma = 0.9
epsilon = 1
epsilon_decay = .99
epsilon_min = 0.1
m.compile(loss='mse',
          optimizer=keras.optimizers.Adadelta())


def remember(state, action, reward, next_state, done):
    global memory
    memory.append([state, action, reward, next_state, done])
    if len(memory) > 1200:
        memory = memory[1:]


def train():
    for i in range(100):
        pool = numpy.zeros((1, 80, 80, 9))
        img, hp1, hp2 = game_utils.fetch_screen()
        pool[:, :, :, 8] = numpy.asarray(img.convert("L").resize((80, 80))).reshape(1, 80, 80)
        state = copy.deepcopy(pool)
        last_hp1 = [hp1, hp1, hp1, hp1, hp1]
        last_hp2 = [hp2, hp2, hp2, hp2, hp2]
        for time_t in range(60):
            # turn this on if you want to render
            # env.render()
            # 选择行为
            action = act(state)
            # 在环境中施加行为推动游戏进行
            img, hp1, hp2 = game_utils.fetch_screen()
            pool[:, :, :, 0: 8] = pool[:, :, :, 1: 9]
            pool[:, :, :, 8] = numpy.asarray(img.convert("L").resize((80, 80))).reshape(1, 80, 80)
            next_state = copy.deepcopy(pool)
            last_hp1.append(hp1)
            last_hp1 = last_hp1[1:]
            last_hp2.append(hp2)
            last_hp2 = last_hp2[1:]
            reward = last_hp2[0] - hp2 - (last_hp1[0] - hp1)
            # 记忆先前的状态，行为，回报与下一个状态
            remember(state, action, reward, next_state, False)
            # 使下一个状态成为下一帧的新状态
            state = copy.deepcopy(next_state)
        # 通过之前的经验训练模型
        game_utils.press_key([0x01])
        replay(32)
        game_utils.press_key([0x01])


def act(state):
    if numpy.random.rand() <= epsilon:
        game_utils.act(int(8 * numpy.random.rand()))
    act_values = m.predict(state)
    game_utils.act(numpy.argmax(act_values[0]))


def replay(batch_size):
    global epsilon
    batches = min(batch_size, len(memory))
    batches = numpy.random.choice(len(memory), batches)
    for i in batches:
        state, action, reward, next_state, done = tuple(memory[i])
        target = reward
        if not done:
            target = reward + gamma * numpy.amax(m.predict(next_state)[0])
        target_f = m.predict(state)
        target_f[0][action] = target
        m.fit(state, target_f, epochs=1, verbose=0)
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay
