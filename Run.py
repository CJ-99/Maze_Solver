import numpy as np
import matplotlib.pyplot as plt
from Environment import MazeEnv
from read_maze import load_maze
from read_maze import get_local_maze_information

load_maze()
agent = MazeEnv(x_in = 1, y_in = 1, x_target = 199, y_target = 199)
Q = np.zeros((201,201,4))
epsilon=0.3
gamma = 0.9
lr = 0.1
wall = 10
time = 1
target = 100
steps=[]
for i in range(15):
    agent.reset()
    step_id=[]
    action = agent.action_current_state(Q, epsilon)
    while True:
        # print(agent.x, agent.y, Q[agent.x][agent.y])
        # print(agent.x, agent.y)
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


fast_actions=[]
fast_local_info=[]
fast_positions=[]

agent.reset()
Q=Q
action = agent.action_current_state(Q, epsilon)
while True:
    #Record Relevant Info
    print(agent.x, agent.y)
    fast_actions.append(action)
    fast_local_info.append(get_local_maze_information(agent.x, agent.y))
    fast_positions.append([agent.x, agent.y])

    agent.next_state(action)
    agent.get_reward(wall, time, target)
    Q = agent.Q_Learning(action, Q, lr, gamma)
    next_action = agent.action_next_state(Q, epsilon)
    epsilon = epsilon - 0.001
    agent.move_to_next_state()
    action = next_action
    if agent.x == agent.x_target and agent.y == agent.y_target:
        print('Finished')
        break


fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(range(15), steps)
ax.set_title('Maze Solver_' + 'Q_Learning' + 'Training')
ax.set_xlabel('Episodes')
ax.set_ylabel('Number of Steps')
ax.grid(True)
plt.show()
plt.savefig('maze_result.png')

np.save('fast_positions.npy', np.array(fast_positions))
np.save('fast_actions.npy', np.array(fast_actions))
np.save('fast_local_info.npy', np.array(fast_local_info))

