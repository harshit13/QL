# -*- coding: utf-8 -*-
# @Author: harshit
# @Date:   2018-09-03 07:41:17
# @Last Modified by:   harshit
# @Last Modified time: 2019-04-29 12:47:44

import numpy as np
import pygame
from VARS import *


class Agent(object):
    """docstring for Agent"""

    def __init__(self, y_pos, score, state):
        super(Agent, self).__init__()
        self.x_pos = 800 * 0.1
        self.y_pos = y_pos
        self.score = score
        self.state = state
        self.actions = [-7, 0]
        self.current_action = 1
        self.reward = 0
        if DISPLAY:
            self.copter = pygame.image.load('helicopter.png')
            self.copter = pygame.transform.scale(self.copter, (100, 50))
        else:
            self.copter = None

    def update(self):
        # self.score += 1
        new_y_pos = self.take_action(USER_INPUT) + 2
        self.y_pos = new_y_pos

    def take_action(self, action=RANDOM):
        # take action greedily
        if action == GREEDY:
            return self.y_pos
        # take action with user input
        elif action == USER_INPUT:
            for event in pygame.event.get():
                # print(event)
                if event.type == pygame.QUIT:
                    pygame.quit()
                if event.type == pygame.KEYDOWN and event.key == pygame.K_UP:
                    self.current_action = 0
                else:
                    self.current_action = 1
            return self.y_pos + self.actions[self.current_action]

        # take action randomly
        else:
            ind = np.random.randint(0, 2)
            return self.y_pos + self.actions[ind]

    def new_reward(self, reward):
        self.reward = reward
        self.score += reward

    def start(self):
        self.score = 0
        self.y_pos = 800 * 0.3
        self.reward = 0
