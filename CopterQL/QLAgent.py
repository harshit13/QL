# -*- coding: utf-8 -*-
# @Author: harshit
# @Date:   2019-04-27 19:21:07
# @Last Modified by:   harshit
# @Last Modified time: 2019-05-01 02:02:29

import numpy as np
from Agent import Agent
from DQN import *
import os
from keras.models import load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
# earlystopper = EarlyStopping(patience=5, verbose=1)
# checkpointer = ModelCheckpoint(model_name, verbose=1, save_best_only=True)
from sklearn.model_selection import KFold

from Queue import ExperienceReplay
import gc


class QLAgent(Agent):
    """docstring for QLAgent"""

    def __init__(self, y_pos, score, state, alpha, gamma):
        super(QLAgent, self).__init__(y_pos, score, state)
        # set learning rate
        self.alpha = alpha
        # set gamma
        self.gamma = gamma
        # the q approximation
        self.model = None
        # the current reward
        self.reward = 0
        # epsilon for greedy action
        self.epsilon = 0.87
        # the q approx model
        if os.path.isfile('model_300_NN2_3.h5'):
            self.model = load_model('model_300_NN2_3.h5')
        else:
            self.model = NN2()

    def getQ(self, state, action):
        inp = state.process_input(action)
        # print(inp.shape)
        return self.model.predict(inp)

    def maxQ(self, state):
        return np.max(
            [self.getQ(state, action) for action in self.actions])

    def updateQ(self, state, action, reward, next_state, isRandom=True):
        # currQ = self.getQ(state, action)
        # print("Action :", action)
        epoch = 1
        if isRandom:
            epoch = 3
        if reward == -10:
            epoch = 10
            print("Training negative reward !!")
        elif reward == 1:
            epoch = 6
            print("Training positive reward !!")

        update = reward
        if reward is not -10:
            update += self.gamma * self.maxQ(next_state)
        inp = state.process_input(action)
        self.model.fit(inp, update.reshape(1, 1), epochs=epoch, verbose=2)
        # self.model.save('model.h5')

    def take_action(self, state):
        # take greedy action
        if np.random.random() > self.epsilon:
            # print("Greedy Action")
            return self.actions[np.argmax([
                self.getQ(state, action) for action in self.actions])], False
        else:
            """if self.epsilon > 0.3:
                self.epsilon *= 0.999999"""
            # print("Random Action")
            return self.actions[np.random.randint(0, 2)], True

    """
    def next_state(self, state, new_action):
        next_state = copy.deepcopy(state)
        next_state.A_y_pos += new_action + 2
        next_state.W1_x_pos -= 5
        if next_state.W1_x_pos < 0:
            next_state.W1_height = 0
        next_state.W2_x_pos -= 5
        next_state.W3_x_pos -= 5
        next_state.W4_x_pos -= 5

        return next_state
        """

    def update(self):
        next_action, isRandom = self.take_action(self.state)
        next_state = self.state.next_state(next_action)
        if isRandom:
            print("Random Action :", next_action)
        else:
            print("Greedy Action :", next_action)
        self.updateQ(
            self.state, next_action, self.reward, next_state, isRandom)
        self.y_pos += next_action + 2


class QAgent(Agent):
    """docstring for QAgent"""

    def __init__(self, y_pos, score, state, model_path):
        super(QAgent, self).__init__(y_pos, score, state)
        self.reward = 0
        self.model = load_model(model_path)

    def take_action(self, state):
        # take action as per policy
        # if np.random.random() > 0.6:
        return self.actions[np.argmax([
            self.getQ(state, a) for a in self.actions])]
        """else:
            return self.actions[np.random.randint(0, 2)]"""

    def getQ(self, state, action):
        return self.model.predict(state.process_input(action))

    def update(self):
        action = self.take_action(self.state)
        self.y_pos += action + 2


class QLAgentEReplay(QLAgent):
    """docstring for QLAgentEReplay"""

    def __init__(self, y_pos, score, state, alpha, gamma, model_name, size):
        super(QLAgentEReplay, self).__init__(y_pos, score, state, alpha, gamma)
        self.buffer = ExperienceReplay(size)
        self.model_name = model_name
        if os.path.isfile(model_name):
            self.model = load_model(model_name)
        else:
            self.model = NN3()
        self.reward = 0
        self.epsilon = 0.87

    def update(self):
        next_action, isRandom = self.take_action(self.state)
        next_state = self.state.next_state(next_action)
        if isRandom:
            print("Random Action :", next_action)
        else:
            print("Greedy Action :", next_action)
        self.add_sample(self.state, next_action, self.reward, next_state)
        self.y_pos += next_action + 2

    def add_sample(self, state, action, reward, next_state):
        update = reward
        if reward is not -10:
            update += self.gamma * self.maxQ(next_state)
        x = state.process_input(action)
        y = update.reshape(1, 1)
        self.buffer.add_new(x, y)
        if len(self.buffer.X) == self.buffer.size:
            # starting K-FOLD training
            folds = KFold(n_splits=5, random_state=2319)

            # processing the data into numpy arrays
            X = np.array(self.buffer.X).reshape(
                (self.buffer.size, NUM_FEATURES))
            Y = np.array(self.buffer.Y).reshape((self.buffer.size, 1))

            # setting model callbacks
            earlystopper = EarlyStopping(patience=5, verbose=2)
            checkpointer = ModelCheckpoint(
                self.model_name, verbose=2, save_best_only=True)

            # starting the iterations
            for fold_, (trn_idx, val_idx) in enumerate(folds.split(X, Y)):
                # Processing train and validation sets
                X_tr, Y_tr = X[trn_idx], Y[trn_idx]
                X_val, Y_val = X[val_idx], Y[val_idx]

                print("Fold:", fold_ + 1, "Training")
                self.model.fit(
                    X_tr, Y_tr,
                    validation_data=(X_val, Y_val),
                    verbose=2, epochs=40, batch_size=400,
                    callbacks=[earlystopper, checkpointer])
            self.buffer.clear()
            gc.collect()
