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
import time


save_data_path = "FXTZ.dat"
gamma = 0.9
epsilon = 1
epsilon_decay = .99
epsilon_min = 0.1

memory = []
try:
    print("Saved Model is Found. Loading...")
    m = keras.models.load_model(save_data_path)
except OSError:
    print("Saved Model is Not Found. Building Model...")
    m = model.get_model()
m.compile(loss='mse',
          optimizer=keras.optimizers.Adadelta())
m.summary()
print("Model Compiled Successfully.")


def remember(state, action, reward, next_state, done):
    global memory
    memory.append([state, action, reward, next_state, done])
    if len(memory) > 1200:
        memory = memory[1:]


def train():
    for i in range(100):
        pool = numpy.zeros((1, 5, 80, 80, 3))
        img, hp1, hp2 = game_utils.fetch_screen()
        pool[0, 0, :, :, :] = numpy.asarray(img.resize((80, 80)))
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
            pool[0, 0: 4, :, :, :] = pool[0, 1: 5, :, :, :]
            pool[0, 0, :, :, :] = numpy.asarray(img.resize((80, 80)))
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
            if hp1 < 13:
                while hp1 < 250:
                    img, hp1, hp2 = game_utils.fetch_screen()
                    game_utils.press_key([0x2C])  # Z
                    time.sleep(0.2)

        # 通过之前的经验训练模型
        game_utils.press_key([0x01])  # Esc
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
    print("Saving Model Data...")
    m.save(save_data_path)
    print("Model Data Saved.")
