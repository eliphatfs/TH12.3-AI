# -*- coding: utf-8 -*-
"""
Created on Wed Aug 15 13:14:09 2018

@author: 北海若
"""

import keras
from keras import layers
import rl.core
import rl.memory
import rl.policy
import rl.agents.dqn
import rl.callbacks
import copy
from rl.callbacks import (
    CallbackList,
    TrainEpisodeLogger,
    TrainIntervalLogger,
    Visualizer
)
import subprocess
import socket
import numpy
import argparse
import sys
import time
import dx_keycode as dxk


EXE_PATH = r"D:\AI_DataSet\TH123\th123\th123.exe"
EVAL_EXE_PATH = r"D:\SRX_UNLOCKED\Ths\[th123] 东方非想天则 (汉化版)\th123_beta.exe"
DUMMY_PATH = r"D:\AI_DataSet\TH123\th123\target_dummy.rep"
IP_PORT = ('127.0.0.1', 5211)
MODEL_SAVE_PATH = r"D:\FXTZ.DQN.GRU.%d.dat"
LOG_PATH = r"dql_training.log"
BATCH = 32

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


def attention_3d_block(inputs):
    a = layers.Permute((2, 1))(inputs)
    a = layers.Dense(30, activation='softmax')(a)
    a_probs = layers.Permute((2, 1))(a)
    output_attention_mul = layers.Multiply()([inputs, a_probs])
    return output_attention_mul


class TH123DllTrainEnv(rl.core.Env):

    def __init__(self):
        self.proc_handle = None
        self.connection = None
        self.socket = socket.socket()
        self.socket.bind(IP_PORT)
        self.socket.listen(8)

        self.first = False
        self.current_act = 0
        self.cache_act = [0, 0]
        self.cache_state = []

    def parse_socket(self, socket_data):
        return [float(x) for x in socket_data.decode().split(" ")][:-1]

    def reset(self):
        stinfo = subprocess.STARTUPINFO()
        stinfo.dwFlags = subprocess.STARTF_USESHOWWINDOW
        stinfo.wShowWindow = subprocess.SW_HIDE
        if (self.proc_handle):
            self.proc_handle.terminate()
        self.connection = None
        fp = open(DUMMY_PATH, mode='rb')
        rep_bytes = list(fp.read(-1))
        fp.close()
        rep_bytes[14] = numpy.random.randint(0, 0x14)
        rep_bytes[63] = numpy.random.randint(0, 0x14)
        fp = open(DUMMY_PATH, mode='wb')
        fp.write(bytes(rep_bytes))
        fp.close()
        self.proc_handle = subprocess.Popen([EXE_PATH,
                                             DUMMY_PATH],
                                            startupinfo=stinfo)
        self.connection, addr = self.socket.accept()
        self.connection.send(b"0 0")
        self.first = True
        self.current_act = 0
        self.cache_state = self.parse_socket(self.connection.recv(255))
        return self.cache_state

    def step(self, action1, action2):
        self.cache_act = [action1, action2]
        end = False
        try:
            self.connection.send(("%d %d" % (self.cache_act[0],
                                             self.cache_act[1]))
                                 .encode())
            old_state = self.cache_state
            self.cache_state = self.parse_socket(self.connection.recv(255))
        except Exception:
            end = True
        my_hp = self.cache_state[self.current_act + 8]
        en_hp = self.cache_state[9 - self.current_act]
        if end:
            rwd = my_hp + 10.0 if int(en_hp) == 0 else 0.0
            rwd = -en_hp - 10.0 if int(my_hp) == 0 else 0.0
        elif old_state[9 - self.current_act] > en_hp:
            rwd = 1.0 + (old_state[9 - self.current_act] - en_hp) / 10.0
        elif old_state[self.current_act + 8] > my_hp:
            rwd = -(0.5 + (old_state[self.current_act + 8] - my_hp) / 20.0)
        else:
            rwd = (my_hp - en_hp) / 300.0
        return (self.cache_state,
                rwd,
                end,
                {})

    def render(self, mode="human", close=False):
        pass

    def close(self):
        if (self.proc_handle):
            self.proc_handle.terminate()

    def new_model(self):
        inp = layers.Input(shape=[30, 11])
        inp_a = attention_3d_block(inp)
        first = conv1d_block(24, 4, padding='causal')(inp_a)
        A, B = wavenet_block(32, 2, 1)(first)
        skip_connections = [B]
        for i in range(1, 3):
            A, B = wavenet_block(32, 2, 2 ** (i % 4))(A)
            skip_connections.append(B)
        for i in range(0, 6):
            A, B = wavenet_block(32, 2, 2 ** (i % 3))(A)
            skip_connections.append(B)
        net = layers.Add()(skip_connections)
        net = layers.LeakyReLU()(net)
        net = conv1d_block(4, 1)(net)
        net = layers.LeakyReLU()(net)
        net = layers.Flatten()(net)
        net = layers.Dense(45, activation='linear')(net)
        return keras.models.Model(inputs=inp,
                                  outputs=net)

    def fit(self, agt1, agt2, env, nb_steps, action_repetition=1,
            callbacks=None, verbose=1,
            visualize=False, nb_max_start_steps=0,
            start_step_policy=None, log_interval=10000,
            save_interval=5000,
            nb_max_episode_steps=None):
        agt1.training = True
        agt2.training = True

        callbacks = [] if not callbacks else callbacks[:]

        if verbose == 1:
            callbacks += [TrainIntervalLogger(interval=log_interval)]
        elif verbose > 1:
            callbacks += [TrainEpisodeLogger()]
        if visualize:
            callbacks += [Visualizer()]
        history = keras.callbacks.History()
        callbacks += [history]
        callbacks = CallbackList(callbacks)
        if hasattr(callbacks, 'set_model'):
            callbacks.set_model(agt1)
            callbacks.set_model(agt2)
        else:
            callbacks._set_model(agt1)
            callbacks._set_model(agt2)
        callbacks._set_env(env)
        params = {
            'nb_steps': nb_steps,
        }
        if hasattr(callbacks, 'set_params'):
            callbacks.set_params(params)
        else:
            callbacks._set_params(params)
        agt1._on_train_begin()
        agt2._on_train_begin()
        callbacks.on_train_begin()

        np = numpy

        episode = np.int16(0)
        agt1.step = np.int16(0)
        agt2.step = np.int16(0)
        observation = None
        episode_reward1 = None
        episode_reward2 = None
        episode_step = None
        did_abort = False
        try:
            while agt1.step < nb_steps:
                if observation is None:  # start of a new episode
                    callbacks.on_episode_begin(episode)
                    episode_step = np.int16(0)
                    episode_reward1 = np.float32(0)
                    episode_reward2 = np.float32(0)

                    agt1.reset_states()
                    agt2.reset_states()
                    observation = copy.deepcopy(env.reset())
                    assert observation is not None

                    if nb_max_start_steps == 0:
                        nb_random_start_steps = 0
                    else:
                        nms = nb_max_start_steps
                        nb_random_start_steps = np.random.randint(nms)
                    for _ in range(nb_random_start_steps):
                        if start_step_policy is None:
                            action1 = env.action_space.sample()
                            action2 = env.action_space.sample()
                        else:
                            action1 = start_step_policy(observation)
                            action2 = start_step_policy(observation)
                        callbacks.on_action_begin(action1)
                        observation, reward, done, info = env.step(action1,
                                                                   action2)
                        observation = copy.deepcopy(observation)
                        callbacks.on_action_end(action1)
                        if done:
                            observation = copy.deepcopy(env.reset())
                            break

                # At this point, we expect to be fully initialized.
                assert episode_reward1 is not None
                assert episode_reward2 is not None
                assert episode_step is not None
                assert observation is not None

                callbacks.on_step_begin(episode_step)
                action1 = agt1.forward(observation)
                action2 = agt2.forward(observation)
                reward = np.float32(0)
                accumulated_info = {}
                done = False
                for _ in range(action_repetition):
                    callbacks.on_action_begin(action1)
                    observation, r, done, info = env.step(action1, action2)
                    observation = copy.deepcopy(observation)
                    for key, value in info.items():
                        if not np.isreal(value):
                            continue
                        if key not in accumulated_info:
                            accumulated_info[key] = np.zeros_like(value)
                        accumulated_info[key] += value
                    callbacks.on_action_end(action1)
                    reward += r
                    if done:
                        break
                if nb_max_episode_steps:
                    if episode_step >= nb_max_episode_steps - 1:
                        # Force a terminal state.
                        done = True
                metrics1 = agt1.backward(reward, terminal=done)
                metrics2 = agt2.backward(-reward, terminal=done)
                episode_reward1 += reward
                episode_reward2 -= reward

                step_logs = {
                    'action1': action1,
                    'action2': action2,
                    'observation': observation,
                    'reward': reward,
                    'metrics': [metrics1[i] + metrics2[i]
                                for i in range(len(metrics1))],
                    'episode': episode,
                    'info': accumulated_info,
                }
                callbacks.on_step_end(episode_step, step_logs)
                episode_step += 1
                agt1.step += 1
                agt2.step += 1
                if agt1.step % save_interval == 0:
                    self.smod1.save_weights(MODEL_SAVE_PATH % 1)
                    self.smod2.save_weights(MODEL_SAVE_PATH % 2)

                if done:
                    agt1.forward(observation)
                    agt2.forward(observation)
                    agt1.backward(0., terminal=False)
                    agt2.backward(0., terminal=False)

                    # This episode is finished, report and reset.
                    episode_logs = {
                        'episode_reward': episode_reward1,
                        'episode_reward1': episode_reward1,
                        'episode_reward2': episode_reward2,
                        'nb_episode_steps': episode_step,
                        'nb_steps': agt1.step,
                    }
                    callbacks.on_episode_end(episode, episode_logs)

                    episode += 1
                    observation = None
                    episode_step = None
                    episode_reward1 = None
                    episode_reward2 = None
        except KeyboardInterrupt:
            did_abort = True
        callbacks.on_train_end(logs={'did_abort': did_abort})
        agt1._on_train_end()
        agt2._on_train_end()

        return history

    def train(self):
        self.smem1 = rl.memory.SequentialMemory(100000, window_length=30)
        self.smem2 = rl.memory.SequentialMemory(100000, window_length=30)
        wrapped_policy = rl.policy.EpsGreedyQPolicy()
        self.spol = rl.policy.LinearAnnealedPolicy(wrapped_policy,
                                                   "eps",
                                                   0.99,
                                                   0.1,
                                                   0.05,
                                                   1200000)
        self.smod1 = self.new_model()
        self.smod1.summary()
        self.smod2 = self.new_model()
        self.smod2.summary()
        try:
            self.smod1.load_weights(MODEL_SAVE_PATH)
        except Exception:
            pass
        dq1 = rl.agents.dqn.DQNAgent(model=self.smod1,
                                     batch_size=BATCH,
                                     nb_actions=45,
                                     policy=self.spol,
                                     memory=self.smem1,
                                     nb_steps_warmup=10000,
                                     gamma=0.998,
                                     target_model_update=50000,
                                     train_interval=4)
        dq2 = rl.agents.dqn.DQNAgent(model=self.smod2,
                                     batch_size=BATCH,
                                     nb_actions=45,
                                     policy=self.spol,
                                     memory=self.smem2,
                                     nb_steps_warmup=10000,
                                     gamma=0.998,
                                     target_model_update=50000,
                                     train_interval=4)
        dq1.compile(keras.optimizers.Adam(), metrics=['mae'])
        dq2.compile(keras.optimizers.Adam(), metrics=['mae'])

        self.fit(dq1, dq2, env,
                 callbacks=[rl.callbacks.FileLogger(LOG_PATH, interval=1)],
                 nb_steps=200000000,
                 log_interval=1000,
                 action_repetition=6)


class ModelCheckpoint(rl.callbacks.Callback):

    def __init__(self):
        self.total_steps = 0

    def on_step_end(self, step, logs={}):
        self.total_steps += 1
        if self.total_steps % 5000 != 0:
            return

        self.model.save_weights(MODEL_SAVE_PATH, overwrite=True)


class TH123EvalEnv(TH123DllTrainEnv):

    def __init__(self, controlling=0):
        self.my = controlling
        self.time = 0
        th123 = subprocess.Popen([EVAL_EXE_PATH])
        time.sleep(5)
        import game_utils as gu
        gu.update_proc_with_pid(th123.pid)

    def act(self, result):
        import game_utils as gu
        my = self.my
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
        for i in range(8):
            if (not oldkeystate[i]) and keystate[i]:
                gu.PressKey(keysetting[my][i])
            if (not keystate[i]) and oldkeystate[i]:
                gu.ReleaseKey(keysetting[my][i])

    def key_to_category(self, key):
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
        return cjkld * 9 + cad * 3 + cws

    def step(self, action):
        import game_utils as gu
        self.act(action)
        time.sleep(0.012)
        gu.update_base()
        pos1x, pos2x = gu.fetch_posx()
        pos1y, pos2y = gu.fetch_posy()
        char1, char2 = gu.fetch_char()
        key1, key2 = gu.fetch_operation()
        hp1, hp2 = gu.fetch_hp()
        wid, wcn = gu.fetch_weather()
        print([pos1x, pos2y,
               pos2x, pos2y, char1, char2,
               self.key_to_category(key1), self.key_to_category(key2),
               hp1, hp2, wid])
        return [pos1x, pos2y,
                pos2x, pos2y, char1, char2,
                self.key_to_category(key1), self.key_to_category(key2),
                hp1, hp2, wid], 0.0, False, {}

    def reset(self):
        import game_utils as gu
        while (gu.fetch_status() not in [0x05, 0x0d, 0x0e, 0x08, 0x09]):
            time.sleep(0.2)
        return self.step(0)[0]

    def close(self):
        return

    def play(self, who=0):
        self.smem = rl.memory.SequentialMemory(100000, window_length=30)
        wrapped_policy = rl.policy.EpsGreedyQPolicy()
        self.spol = rl.policy.LinearAnnealedPolicy(wrapped_policy,
                                                   "eps",
                                                   0.99,
                                                   0.1,
                                                   0.99,
                                                   1000000)
        self.smod = self.new_model()
        self.smod.summary()
        try:
            self.smod.load_weights(MODEL_SAVE_PATH % (who + 1))
        except Exception:
            pass
        dqn = rl.agents.dqn.DQNAgent(model=self.smod,
                                     batch_size=BATCH,
                                     nb_actions=45,
                                     policy=self.spol,
                                     memory=self.smem,
                                     nb_steps_warmup=18000,
                                     gamma=0.995,
                                     target_model_update=50000,
                                     train_interval=8)
        dqn.compile(keras.optimizers.Adam())
        dqn.test(self,
                 action_repetition=6)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deep Q Learning for TH12.3")
    parser.add_argument("--train", dest="train", action="store_true",
                        help="Start Train Mode")
    parser.add_argument("--eval", dest="eval", action="store_true",
                        help="Start Evaluation Mode")
    args = parser.parse_args(sys.argv[1:])
    if args.train and args.eval:
        raise ValueError("Both --train and --eval Found in Args.")
    elif args.train:
        env = TH123DllTrainEnv()
        env.train()
    elif args.eval:
        env = TH123EvalEnv()
        env.play()
    else:
        parser.print_help()
