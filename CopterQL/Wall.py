# -*- coding: utf-8 -*-
# @Author: harshit
# @Date:   2018-09-03 07:40:52
# @Last Modified by:   harshit
# @Last Modified time: 2019-04-26 18:44:47


class Wall(object):
    """docstring for Wall"""

    def __init__(self, height, x_pos, x_speed):
        super(Wall, self).__init__()
        self.height = height
        self.x_pos = x_pos
        self.isAlive = True
        self.speed = x_speed

    def update(self):
        self.x_pos = self.x_pos - self.speed
