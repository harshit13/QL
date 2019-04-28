# -*- coding: utf-8 -*-
# @Author: harshit
# @Date:   2019-04-28 06:26:52
# @Last Modified by:   harshit
# @Last Modified time: 2019-04-28 10:00:13

from QLAgent import *
from Agent import *
from Game import *
import copy

# static limits for data
AGENT_Y_POS_MIN, AGENT_Y_POS_RANGE = 5, 495 - 5
WALL_HEIGHT_MIN, WALL_HEIGHT_RANGE = 50, 400 - 50
WALL_X_POS_MIN, WALL_X_POS_RANGE = 0, 800 - 0

# 1 Agent y position
# 2-9 Walls x position and height
# 10-11 one hot encoded action #2
NUM_FEATURES = 11


class State(object):
    """docstring for State"""

    def __init__(self):
        super(State, self).__init__()
        self.A_y_pos = 800 * 0.3
        self.W1_height = 0
        self.W1_x_pos = 800
        self.W2_height = 0
        self.W2_x_pos = 800
        self.W3_height = 0
        self.W3_x_pos = 800
        self.W4_height = 0
        self.W4_x_pos = 800

    def game_update(self, agent, walls):
        self.A_y_pos = agent.y_pos

        if len(walls) > 0:
            self.W1_height = walls[0].height
            self.W1_x_pos = walls[0].x_pos
        else:
            self.W1_height = 0
            self.W1_x_pos = 800

        if len(walls) > 1:
            self.W2_height = walls[1].height
            self.W2_x_pos = walls[1].x_pos
        else:
            self.W2_height = 0
            self.W2_x_pos = 800

        if len(walls) > 2:
            self.W3_height = walls[2].height
            self.W3_x_pos = walls[2].x_pos
        else:
            self.W3_height = 0
            self.W3_x_pos = 800

        if len(walls) > 3:
            self.W4_height = walls[3].height
            self.W4_x_pos = walls[3].x_pos
        else:
            self.W4_height = 0
            self.W4_x_pos = 800

    def next_state(self, action):
        next_S = copy.deepcopy(self)
        next_S.A_y_pos += action + 2
        next_S.W1_x_pos -= 5
        if next_S.W1_x_pos < 0:
            # change W1 to W2
            next_S.W1_x_pos = next_S.W2_x_pos - 5
            next_S.W1_height = next_S.W2_height
            # change W2 to W3
            next_S.W2_x_pos = next_S.W3_x_pos - 5
            next_S.W2_height = next_S.W3_height
            # change W3 to W4
            next_S.W3_x_pos = next_S.W4_x_pos - 5
            next_S.W3_height = next_S.W4_height
            # set W4 height to 0
            next_S.W4_height = 0
            next_S.W4_x_pos = 800
        else:
            next_state.W2_x_pos -= 5
            next_state.W3_x_pos -= 5
            next_state.W4_x_pos -= 5
        return next_S

    def print_state(self):
        print(
            self.A_y_pos,
            self.W1_height, self.W1_x_pos, self.W2_height, self.W2_x_pos,
            self.W3_height, self.W3_x_pos, self.W4_height, self.W4_x_pos)

    def process_input(self, action):
        inp = np.ndarray(shape=(NUM_FEATURES,))
        inp[0] = (self.A_y_pos - AGENT_Y_POS_MIN) / AGENT_Y_POS_RANGE
        inp[1] = (self.W1_height - WALL_HEIGHT_MIN) / WALL_HEIGHT_RANGE
        inp[2] = (self.W1_x_pos - WALL_X_POS_MIN) / WALL_X_POS_RANGE
        inp[3] = (self.W2_height - WALL_HEIGHT_MIN) / WALL_HEIGHT_RANGE
        inp[4] = (self.W2_x_pos - WALL_X_POS_MIN) / WALL_X_POS_RANGE
        inp[5] = (self.W3_height - WALL_HEIGHT_MIN) / WALL_HEIGHT_RANGE
        inp[6] = (self.W3_x_pos - WALL_X_POS_MIN) / WALL_X_POS_RANGE
        inp[7] = (self.W4_height - WALL_HEIGHT_MIN) / WALL_HEIGHT_RANGE
        inp[8] = (self.W4_x_pos - WALL_X_POS_MIN) / WALL_X_POS_RANGE
        inp[9] = 1 if action == -7 else 0
        inp[10] = 1 if action == 0 else 0
        # print(inp.shape)
        return inp.reshape((1, NUM_FEATURES))
