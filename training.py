# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 11:53:16 2018

@author: 北海若
"""

import game_utils
import model
import time
import numpy
import keras
import copy


m = model.get_model()
memory = []
gamma = 0.9
epsilon = 1
epsilon_decay = .995
epsilon_min = 0.1
learning_rate = 0.0001
m.compile(loss='mse',
          optimizer=keras.optimizers.RMSprop(lr=learning_rate))


def remember(state, action, reward, next_state, done):
        memory.append((state, action, reward, next_state, done))


def train():
    for i in range(30):
        img, hp1, hp2 = game_utils.fetch_screen()
        state = numpy.reshape(numpy.asarray(img.resize((80, 80))), [1, 80, 80, 3])
        for time_t in range(600):
            # turn this on if you want to render
            # env.render()
            # 选择行为
            action = act(state)
            time.sleep(0.017)
            # 在环境中施加行为推动游戏进行
            img, hp1, hp2 = game_utils.fetch_screen()
            next_state = numpy.reshape(numpy.asarray(img.resize((80, 80))), [1, 80, 80, 3])
            reward = hp1 - hp2
            # 记忆先前的状态，行为，回报与下一个状态
            remember(state, action, reward, next_state, False)
            # 使下一个状态成为下一帧的新状态
            state = copy.deepcopy(next_state)
        # 通过之前的经验训练模型
        replay(32)


def act(state):
    if numpy.random.rand() <= epsilon:
        game_utils.act(int(6 * numpy.random.rand()))
    act_values = m.predict(state)
    game_utils.act(numpy.argmax(act_values[0]))


def replay(batch_size):
    global epsilon
    batches = min(batch_size, len(memory))
    batches = numpy.random.choice(len(memory), batches)
    for i in batches:
        state, action, reward, next_state, done = memory[i]
        target = reward
        if not done:
            target = reward + gamma * numpy.amax(m.predict(next_state)[0])
        target_f = m.predict(state)
        target_f[0][action] = target
        m.fit(state, target_f, nb_epoch=1, verbose=0)
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay
