# -*- coding: utf-8 -*-
# @Author: harshit
# @Date:   2019-04-27 19:21:07
# @Last Modified by:   harshit
# @Last Modified time: 2019-04-28 10:33:18

import numpy as np
from Agent import *
from State import *
import copy
from DQN import *
import os
from keras.models import load_model

GREEDY, USER_INPUT, RANDOM = 0, 1, 2


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
        self.epsilon = 1
        # the current state
        self.state = state
        # the q approx model
        if os.path.isfile('model.h5'):
            self.model = load_model('model.h5')
        else:
            self.model = NN()

    def getQ(self, state, action):
        inp = state.process_input(action)
        # print(inp.shape)
        return self.model.predict(inp)

    def maxQ(self, state):
        return np.max(
            [self.getQ(state, action) for action in self.actions])

    def updateQ(self, state, action, reward, next_state):
        # currQ = self.getQ(state, action)
        update = reward + self.gamma * self.maxQ(next_state)
        inp = state.process_input(action)
        self.model.fit(inp, update.reshape(1, 1), verbose=0)
        self.model.save('model.h5')

    def take_action(self, state):
        # take greedy action
        if np.random.random() > self.epsilon:
            self.epsilon *= 0.99
            return self.actions[np.argmax([
                self.getQ(state, action) for action in actions])]
        else:
            return self.actions[np.random.randint(0, 2)]

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

    def update(self):
        next_action = self.take_action(self.state)
        next_state = self.next_state(self.state, next_action)
        self.updateQ(self.state, next_action, self.reward, next_state)
        self.y_pos += next_action + 2
