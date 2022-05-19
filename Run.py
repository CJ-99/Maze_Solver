import numpy as np
import matplotlib.pyplot as plt
from Environment import MazeEnv
from read_maze import load_maze
from read_maze import get_local_maze_information

load_maze()
agent = MazeEnv(x_in = 1, y_in = 1, x_target = 199, y_target = 199)
# Q = np.random.rand(201, 201, 4)
Q = np.zeros((201,201,4))
steps=[]
epsilon=0.2
gamma = 0.9
lr = 0.1
wall = 10
time = 1
target = 100
for i in range(1):
    agent.reset()
    step_id=[]
    action = agent.action_current_state(Q, epsilon)
    while True:
        # print(agent.x, agent.y, Q[agent.x][agent.y])
        print(agent.x, agent.y)
        step_id.append(action)
        agent.next_state(action)
        agent.get_reward(wall, time, target)
        Q = agent.Q_Learning(action, Q, lr, gamma)
        next_action = agent.action_next_state(Q, epsilon)
        epsilon = epsilon - 0.001
        agent.move_to_next_state()
        action = next_action
        if agent.x == agent.x_target and agent.y == agent.y_target:
            steps.append(len(step_id))
            print('Finished')
            break

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(range(1), steps)
ax.set_title('Maze Solver_' + 'Q_Learning')
ax.set_xlabel('Episodes')
ax.set_ylabel('Step Number')
ax.grid(True)
plt.show()
# plt.savefig('maze_result.png')
