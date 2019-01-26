"""BattleTank environment"""
import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import mgym
import copy
import pyglet
from collections import deque
import random

GRID_HEIGHT = 30
GRID_WIDTH = 30
GRID_SIZE = 15
MARGIN = 1

BULLET_ID = 2
BULLET_COLOR = np.array([50,50,50])

SPEED = {0: (0, -1),
         1: (0, 1),
         2: (-1, 0),
         3: (1, 0)}
         
BULLET_SPEED = {0: (0, -2),
         1: (0, 2),
         2: (-2, 0),
         3: (2, 0)}

OPPOSITE_DIRECTION = {0: 1,
                      1: 0,
                      2: 3,
                      3: 2}

WALL_ID = 1
WALL_COLOR = np.array([25,25,25])

TANK_ID_MIN = 3
TANK_WIDTH = 5
TANK_HEIGHT = 5
TANK_BARREL_LENGTH = 3
TANK_MIN_COLOR = np.array([100,100,100])
TANK_MAX_COLOR = np.array([255,255,255])
MAX_TANKS = min(GRID_HEIGHT/TANK_HEIGHT, GRID_WIDTH/TANK_WIDTH)

TANK_HIT_SCORE = 10
TANK_DESTROYED_SCORE = -10

class BattleTankEnv(mgym.MEnv):
    """ 

    """

    def __init__(self):
        self.N = None  # set in reset function

    def reset(self, N=2):
        if N>MAX_TANKS: 
            print("There should not be more than %s tanks." % MAX_TANKS)
            return
        self.done = False
        self.N = N
        self.nA = 4
        self.observation_space = spaces.Box(low=0, high=255, shape=(GRID_HEIGHT, GRID_WIDTH, 3), dtype=np.uint8)
        self.action_space = spaces.Tuple(
            [spaces.Discrete(self.nA) for _ in range(self.N)])
        self.grid = np.zeros((GRID_HEIGHT, GRID_WIDTH))
        self.colored_grid = np.zeros((GRID_HEIGHT, GRID_WIDTH,3))
        self.bullets = []
        self.tanks = []
        self.walls = []
        self.scores = []
        self.create_boarder()

        for i in range(N):
            tank_id = TANK_ID_MIN + i
            location = self._get_init_location(tank_id)
            self.tanks.append(Tank(*location, 0, tank_id))
            self.scores.append(0)

        # one tank, one bullet
        for i in range(N):
            tank_id = TANK_ID_MIN + i
            location = self._get_barrel_location(tank_id)
            self.bullets.append(Bullet(*location, self.tanks[i].direction))

        self.update_grid()
        
        return self.colored_grid

    def seed(self, seed=None):
        random.seed(seed)

    def create_boarder(self):
        self.walls = []

        # south wall
        for i in range(GRID_WIDTH):
            self.walls.append(Wall(i, 0))

        # north wall
        for i in range(GRID_WIDTH):
            self.walls.append(Wall(i, GRID_HEIGHT - 1))

        # west wall
        for i in range(1, GRID_HEIGHT - 1):
            self.walls.append(Wall(0, i))

        # east wall
        for i in range(1, GRID_HEIGHT - 1):
            self.walls.append(Wall(GRID_HEIGHT - 1, i))

    def step(self, action):
        for i, tank in enumerate(self.tanks):
            tank.try_update_position(action[i], self.grid)
        
        for i, bullet in enumerate(self.bullets):
            bullet.update_position()
            
        new_scores = [0 for tank in self.tanks]
        # check bullet collision
        for i, bullet in enumerate(self.bullets):
            is_bullet_destroyed = False
            # check if bullet is outside the grid
            if bullet.x < 0 or bullet.x >= GRID_WIDTH or bullet.y < 0 or bullet.y >= GRID_HEIGHT:
                location = self._get_barrel_location(i+TANK_ID_MIN)
                self.bullets[i] = Bullet(*location, self.tanks[i].direction)
                continue
            # check if the bullet hit a tank(not self)
            bullet_at_grid = self.grid[bullet.x, bullet.y]
            if bullet_at_grid >= TANK_ID_MIN and bullet_at_grid != i+TANK_ID_MIN:
                is_bullet_destroyed = True
                self.tanks[bullet_at_grid-TANK_ID_MIN].alive = False
                new_scores[i] = TANK_HIT_SCORE
                new_scores[bullet_at_grid-TANK_ID_MIN] = TANK_DESTROYED_SCORE
                location = self._get_barrel_location(i+TANK_ID_MIN)
                self.bullets[i] = Bullet(*location, self.tanks[i].direction)

        self.remove_dead_tanks()

        if len(self._alive_tanks())<=1:
            self.done = True
#            print('\n')
#            print('*******************')
#            print('**** GAME OVER ****')
#            print('******************* \n')

        self.update_grid()

        rewards = [(new_scores[i] - self.scores[i]) for i in range(self.N)]
        self.scores = new_scores

        return self.colored_grid, rewards, self.done, {}

    def remove_dead_tanks(self):
        pass

    def render(self, mode='human'):
        for row in range(0, GRID_HEIGHT):
            for col in range(0, GRID_WIDTH):
                if self.grid[row, col] == BULLET_ID:
                    color = BULLET_COLOR#'red'
                    self._draw_square(row, col, color)
                elif self.grid[row, col] == WALL_ID:
                    color = WALL_COLOR#'blue'
                    self._draw_square(row, col, color)
                elif self.grid[row, col] >= 3:
                    color = ((self.grid[row, col] - 3) /
                                self.N * (TANK_MAX_COLOR - TANK_MIN_COLOR) + TANK_MIN_COLOR).astype(np.int)
                    self._draw_square(row, col, color)

    def update_grid(self):
        self.grid = np.zeros((GRID_HEIGHT, GRID_WIDTH))
        for tank in self._alive_tanks():
            # TODO: Draw the barrel
            for i in range(TANK_WIDTH):
                for j in range(TANK_HEIGHT):
                    self.grid[tank.x+i, tank.y+j] = tank.id
        for bullet in self.bullets:
            self.grid[bullet.x, bullet.y] = bullet.id
            self.colored_grid[bullet.x, bullet.y] = BULLET_COLOR
        for wall in self.walls:
            self.grid[wall.x, wall.y] = wall.id
            self.colored_grid[wall.x, wall.y] = WALL_COLOR

    def _alive_tanks(self):
        return [tank for tank in self.tanks if tank.alive]

    def _get_init_location(self, tank_id):
        id = tank_id-TANK_ID_MIN
        corner_x, corner_y = int(GRID_WIDTH*id/self.N+MARGIN), int(GRID_HEIGHT*id/self.N+MARGIN)
        return (corner_x, corner_y)
        
    def _get_barrel_location(self, tank_id):
        id = tank_id-TANK_ID_MIN
        tank = self.tanks[id]
        return tank.x + np.sign(tank.xspeed)*TANK_BARREL_LENGTH, tank.y + np.sign(tank.yspeed)*TANK_BARREL_LENGTH

    def _draw_square(self, row, col, color):

        if isinstance(color, int):
            RGB = (color, color, color)
        elif isinstance(color, str):
            if color == 'red':
                RGB = (255, 0, 0)
            elif color == 'green':
                RGB = (0, 255, 0)
            elif color == 'blue':
                RGB = (0, 0, 255)
        else:
            RGB = (color[0], color[1], color[2])

        square_coords = (row * GRID_SIZE + MARGIN, col * GRID_SIZE + MARGIN,
                         row * GRID_SIZE + MARGIN, col * GRID_SIZE + GRID_SIZE - MARGIN,
                         row * GRID_SIZE + GRID_SIZE - MARGIN, col * GRID_SIZE + MARGIN,
                         row * GRID_SIZE + GRID_SIZE - MARGIN, col * GRID_SIZE + GRID_SIZE - MARGIN)
        pyglet.graphics.draw_indexed(4, pyglet.gl.GL_TRIANGLES,
                                     [0, 1, 2, 1, 2, 3],
                                     ('v2i', square_coords),
                                     ('c3B', (*RGB, *RGB, *RGB, *RGB)))


class Bullet(object):
    def __init__(self, x, y, direction):
        self.x = x
        self.y = y
        self.id = BULLET_ID
        self.direction = direction
        self.xspeed = SPEED[self.direction][0]
        self.yspeed = SPEED[self.direction][1]
        
    def update_position(self):
        self.x = self.x + self.xspeed
        self.y = self.y + self.yspeed


class Wall(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.id = WALL_ID


class Tank(object):
    def __init__(self, x, y, direction, id):
        #upper left
        self.x = x
        self.y = y

        self.direction = direction
        self.xspeed = SPEED[self.direction][0]
        self.yspeed = SPEED[self.direction][1]
        self.id = id
        self.alive = True

    def try_update_position(self, action, grid):
        if not self.alive: return False
        self.direction = action
        self.xspeed = SPEED[self.direction][0]
        self.yspeed = SPEED[self.direction][1]
        x = self.x + self.xspeed
        y = self.y + self.yspeed
        if x < 0 or x+TANK_WIDTH > GRID_WIDTH or y < 0 or y+TANK_HEIGHT > GRID_HEIGHT: return False
        for i in range(TANK_WIDTH):
            for j in range(TANK_HEIGHT):
                if grid[i+x,j+y] != 0 and grid[i+x,j+y] != BULLET_ID and grid[i+x,j+y] != self.id: return False
        # update position
        self.x = x
        self.y = y

    def update_position(self, action):
        if self.alive:
            self.direction = action

            self.xspeed = SPEED[self.direction][0]
            self.yspeed = SPEED[self.direction][1]

            self.x = self.x + self.xspeed
            self.y = self.y + self.yspeed
        

class Window(pyglet.window.Window):
    def __init__(self):
        super().__init__(GRID_HEIGHT * GRID_SIZE, GRID_WIDTH * GRID_SIZE)
        self.env = BattleTankEnv()
        pyglet.clock.schedule_interval(self.update, 1.0 / 8.0)
        self.action = [2, 2]
        self.pause = False

    def on_draw(self):
        self.clear()
        self.env.render()

    def update(self, dt):
        if not self.pause:
            # self.action = self.env.action_space.sample()
            s, r, d, _ = self.env.step(self.action)
            if self.env.done:
                self.close()

    def on_key_press(self, symbol, mod):

        if symbol == pyglet.window.key.S:
            self.action[0] = 0
        if symbol == pyglet.window.key.W:
            self.action[0] = 1
        if symbol == pyglet.window.key.A:
            self.action[0] = 2
        if symbol == pyglet.window.key.D:
            self.action[0] = 3

        if symbol == pyglet.window.key.DOWN:
            self.action[1] = 0
        if symbol == pyglet.window.key.UP:
            self.action[1] = 1
        if symbol == pyglet.window.key.LEFT:
            self.action[1] = 2
        if symbol == pyglet.window.key.RIGHT:
            self.action[1] = 3

        if symbol == pyglet.window.key.SPACE:
            self.pause = not self.pause


if __name__ == "__main__":
    num_tanks = 2
    wind = Window()
    wind.env.reset(num_tanks)
    pyglet.app.run()
