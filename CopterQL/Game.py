# -*- coding: utf-8 -*-
# @Author: harshit
# @Date:   2018-09-03 07:41:06
# @Last Modified by:   harshit
# @Last Modified time: 2019-04-28 10:21:20

from Agent import *
from Wall import *
from State import *
import pygame

black = (0, 0, 0)
white = (255, 255, 255)

DISPLAY = False

class Game(object):
    """docstring for Game"""

    def __init__(self, agent, shape, state):
        super(Game, self).__init__()
        pygame.init()
        self.shape = shape
        self.agent = agent
        self.walls = []
        self.X = 800
        self.Y = 500
        self.top = 5
        self.bottom = 495
        self.display = pygame.display.set_mode(shape)
        self.clock = pygame.time.Clock()
        self.finished = False
        self.state = state

        if DISPLAY:
            # set caption for display
            pygame.display.set_caption('Copter')

    def update(self):
        self.add_new_walls()
        wall_removed = False
        for wall in self.walls:
            wall.update()
            if wall.x_pos < 0:
                wall_removed = True
                wall.isAlive = False
                self.walls.remove(wall)
        if wall_removed:
            self.agent.new_reward(1)
        else:
            self.agent.new_reward(0)
        self.state.game_update(self.agent, self.walls)
        self.finished = self.collision()
        if self.finished:
            self.agent.new_reward(-100)
        elif wall_removed:
            self.agent.new_reward(1)
        else:
            self.agent.new_reward(0)

        self.agent.update()
        # update the display

    def collision(self):
        # iterate through all walls, if agent at some wall
        #   return True
        if self.agent.x_pos + 90 > self.walls[0].x_pos and \
                self.agent.x_pos < self.walls[0].x_pos + 50:
            if self.agent.y_pos + 40 > 500 - self.walls[0].height:
                return True

        if self.agent.y_pos > 495 or self.agent.y_pos < 5:
            return True
        # check if agent.y_pos is in between (5 - 495)
        #   return False else True
        return False

    def add_new_walls(self):
        if len(self.walls) in [1, 2, 3, 4]:
            last_wall_pos = self.walls[-1].x_pos
            if last_wall_pos < self.shape[0] * 0.7:
                wall = self.new_random_wall()
                self.walls.append(wall)
        elif len(self.walls) == 0:
            # randomly add new wall
            wall = self.new_random_wall()
            self.walls.append(wall)
        else:
            pass

    def new_random_wall(self):
        # self.agent.new_reward(1)
        height = np.random.randint(50, 400)
        x_speed = 5
        x_pos = 800
        return Wall(height, x_pos, x_speed)

    def start(self):
        self.agent.start()
        self.walls = []
        self.finished = False

    def display_wall(self, wall):
        width = 80
        pygame.draw.rect(
            self.display, black,
            [wall.x_pos, 500 - wall.height, width, wall.height])

    def display_copter(self):
        self.display.blit(
            self.agent.copter, (self.agent.x_pos, self.agent.y_pos))

    def run(self):
        # pygame.display.set_caption('Copter')
        while not self.finished:
            self.update()
            self.display.fill(white)

            if DISPLAY:
                # draw top and bottom
                pygame.draw.rect(self.display, black, [0, 0, 800, 10])
                pygame.draw.rect(self.display, black, [0, 490, 800, 10])

            # display walls
            for wall in self.walls:
                self.display_wall(wall)

            # display copter
            self.display_copter()

            if DISPLAY:
                # update display
                pygame.display.update()
            self.clock.tick(60)
            print(
                "Reward:", self.agent.reward,
                "Score:", self.agent.score,
                "#Walls:", len(self.walls))
            self.state.print_state()

        # pygame.quit()


def main():
    shape = (800, 500)
    state = State()
    agent = QLAgent(shape[1] * 0.3, 0, state, 0.1, 0.7)
    game = Game(agent, shape, state)
    for s in range(100):
        print('\n\n# Session:', s, '>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        game.start()
        game.run()


if __name__ == '__main__':
    main()
