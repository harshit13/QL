# -*- coding: utf-8 -*-
# @Author: harshit
# @Date:   2019-04-29 12:59:05
# @Last Modified by:   harshit
# @Last Modified time: 2019-04-29 13:39:59


class ExperienceReplay(object):
    """Queue to store all states, action and q tuples"""

    def __init__(self, size):
        super(ExperienceReplay, self).__init__()
        self.size = size
        self.X = []
        self.Y = []

    def add_new(self, x, y):
        self.X.append(x)
        self.Y.append(y)

    def clear(self):
        self.X.clear()
        self.Y.clear()
