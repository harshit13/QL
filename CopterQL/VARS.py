# -*- coding: utf-8 -*-
# @Author: harshit
# @Date:   2019-04-28 14:37:23
# @Last Modified by:   harshit
# @Last Modified time: 2019-05-01 02:30:44

black = (0, 0, 0)
white = (255, 255, 255)
shape = (800, 500)

DISPLAY = True

GREEDY, USER_INPUT, RANDOM = 0, 1, 2

# static limits for data
AGENT_Y_POS_MIN, AGENT_Y_POS_RANGE = 0, 500 - 0
WALL_HEIGHT_MIN, WALL_HEIGHT_RANGE = 0, 500 - 0
WALL_X_POS_MIN, WALL_X_POS_RANGE = 0, 800 - 0

# 1 Agent y position
# 2-9 Walls x position and height
# 10-11 one hot encoded action #2
NUM_FEATURES = 11
