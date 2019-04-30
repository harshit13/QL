# -*- coding: utf-8 -*-
# @Author: harshit
# @Date:   2018-09-25 06:13:43
# @Last Modified by:   harshit
# @Last Modified time: 2018-09-25 06:59:53
# start of game

# import numpy as np


class Agent():
    """Agent"""

    def __init__(self, size, actions, pos):
        # super(Agent, self).__init__()
        self.size = size
        self.actions = actions
        self.pos = pos

    def get_next_action(self, curr, game_map):
        inp = int(input('Select from 0-7 : '))
        c = self.pos + self.actions[inp]
        while c[0] < 0 or c[1] < 0 or \
                c[0] >= self.size[0] or c[1] >= self.size[1]:
            print(c[0], c[1], "Enter again ")
            inp = int(input('Select from 0-7 : '))
            c = self.pos + self.actions[inp]
        return self.actions[inp]

    def take_action(self, action):
        # take the required action
        self.pos[0] += action[0]
        self.pos[1] += action[1]
        return self.pos
