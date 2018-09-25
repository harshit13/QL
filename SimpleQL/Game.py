# -*- coding: utf-8 -*-
# @Author: harshit
# @Date:   2018-09-24 18:59:13
# @Last Modified by:   harshit
# @Last Modified time: 2018-09-25 06:59:44
# start of game

import numpy as np
from Agent import Agent


class Game():
    """docstring for Game"""

    def __init__(self, size, agent, game_map):
        # super(Game, self).__init__()
        self.map = game_map
        self.size = size
        self.agent = agent

    def run(self):
        # to run the program
        curr = np.zeros(shape=(2,), dtype=np.int8)

        while True:
            self.printMap()
            action = self.agent.get_next_action(curr, self.map)

            curr = self.agent.take_action(action)

            if self.map[curr[0]][curr[1]] == 'G':
                # take award as +1
                print("Reached Goal")
                break
            elif self.map[curr[0]][curr[1]] == 'X':
                # take negative reward
                print("Reached Gutter")
                break

        print("Game Over")

    def printMap(self):
        for i in range(self.size[0]):
            for j in range(self.size[1]):
                if i == self.agent.pos[0] and j == self.agent.pos[1]:
                    print('A', end=" ")
                else:
                    print(self.map[i][j], end=" ")
            print()


if __name__ == '__main__':
    size = np.array([6, 6], dtype=np.int8)
    game_map = np.array(
        [
            ['S', '+', '+', '+', 'X', '+'],
            ['+', '+', 'X', '+', '+', '+'],
            ['X', '+', '+', '+', 'X', '+'],
            ['+', '+', '+', '+', 'X', '+'],
            ['+', 'X', '+', '+', 'X', '+'],
            ['+', '+', 'X', '+', '+', 'G']
        ],
        dtype=np.object
    )

    actions = np.array(
        [
            [1, 1], [1, 0], [1, -1], [0, -1],
            [-1, -1], [-1, 0], [-1, 1], [0, 1]
        ],
        dtype=np.int8
    )

    agent = Agent(size, actions, np.array([0, 0], dtype=np.int8))
    game = Game(size, agent, game_map)
    game.run()
