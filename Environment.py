import numpy as np
import read_maze
from read_maze import load_maze
from read_maze import get_local_maze_information
from read_maze import maze_cells
import pandas as pd


class MazeEnv:
    def __init__(self, x_in = 1, y_in = 1, x_target = 199, y_target=199):
        self.maze = load_maze()
        self.x_in, self.y_in = x_in, y_in
        self.x_target, self.y_target = x_target, y_target

    def reset(self):
        self.x, self.y = self.x_in, self.y_in

    def action_current_state(self, Q, epsilon):
        if np.random.uniform() < epsilon:
            action = np.random.randint(4)
        else:
            action = np.argmax(Q[self.x][self.y], axis=0)
        return action

    def action_next_state(self, Q, epsilon):
        if np.random.uniform() < epsilon:
            action = np.random.randint(4)
        else:
            action = np.argmax(Q[self.x_next][self.y_next], axis=0)
        return action

    def next_state(self, action):
        directions = ['u', 'r', 'd', 'l']
        dir = directions[action]

        if dir == 'u':
            if get_local_maze_information(self.x, self.y)[0][1][0] == 0: #can't move up
                self.x_next, self.y_next = self.x, self.y #stay in place
                # print('wall_up')
            elif get_local_maze_information(self.x, self.y)[0][1][1] != 0: #fire in place
                self.x_next, self.y_next = self.x, self.y #stay in place
            else:
                # print("move: ", self.x, self.y, get_local_maze_information(self.x, self.y))
                self.x_next, self.y_next = self.x-1, self.y

        if dir == 'r':
            if get_local_maze_information(self.x, self.y)[1][2][0] == 0:
                self.x_next, self.y_next = self.x, self.y #stay in place
                # print('wall_right')
            elif get_local_maze_information(self.x, self.y)[1][2][1] != 0: #fire in place
                self.x_next, self.y_next = self.x, self.y #stay in place
            else:
                # print("move: ", self.x, self.y, get_local_maze_information(self.x, self.y))
                self.x_next, self.y_next = self.x, self.y+1

        if dir == 'd':
            if get_local_maze_information(self.x, self.y)[2][1][0] == 0:
                self.x_next, self.y_next = self.x, self.y #stay in place
                # print('wall_down')
            elif get_local_maze_information(self.x, self.y)[2][1][1] != 0: #fire in place
                self.x_next, self.y_next = self.x, self.y #stay in place
            else:
                # print("move: ", self.x, self.y, get_local_maze_information(self.x, self.y))
                self.x_next, self.y_next = self.x+1, self.y

        if dir == 'l':
            if get_local_maze_information(self.x, self.y)[1][0][0] == 0:
                self.x_next, self.y_next = self.x, self.y #stay in place
                # print('wall_left')
            elif get_local_maze_information(self.x, self.y)[1][0][1] != 0: #fire in place
                self.x_next, self.y_next = self.x, self.y #stay in place
            else:
                # print("move: ", self.x, self.y, get_local_maze_information(self.x, self.y))
                self.x_next, self.y_next = self.x, self.y-1

    def get_reward(self, wall, time, target):
        self.reward = 0
        if get_local_maze_information(self.x_next, self.y_next)[1][1][0] == 0:
                    self.reward -= wall
        elif self.x_next == self.x_target and self.y_next == self.y_target:
            self.reward += target
        self.reward -= time

    def Q_Learning(self, action, Q, lr, gamma):
        #Q[s,a] = Q[s,a] + lr*(r + gamma * max(Q[s_,:]) - Q[s,a])
        if self.x_next == self.x_target and self.y_next ==  self.y_target:
            Q[self.x][self.y][action] = Q[self.x][self.y][action] + lr*(self.reward - Q[self.x][self.y][action])
        else:
            Q[self.x][self.y][action] = Q[self.x][self.y][action] + lr * (self.reward + gamma * np.max(Q[self.x_next][self.y_next], axis=0) - Q[self.x][self.y][action])
        return Q

    def move_to_next_state(self):
        self.x, self.y = self.x_next, self.y_next

