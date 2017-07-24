import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class Agent:
    def __init__(self, pos, velo):
        self.pos = pos
        self.velo_temp = velo
        self.velo = velo


x_list = []
y_list = []
agents = []
w = 0.5
N = 5


def initialize():
    for i in range(N):
        agents.append(Agent(2 * np.random.random(2), 2 * np.random.random(2)))


def find_nearest_neighbor(aj):
    distances = [np.linalg.norm(aj.velo - ai.velo) for ai in agents]
    distances[distances.index(0.0)] = max(distances)
    return (agents[distances.index(min(distances))])


def get_x_pos():
    ret_list = []
    for aj in agents:
        ret_list.append(aj.pos[0])
    return ret_list


def get_y_pos():
    ret_list = []
    for aj in agents:
        ret_list.append(aj.pos[1])
    return ret_list


initialize()
print(find_nearest_neighbor(agents[0]))

test_x = [[1, 2, 3], [2, 3, 4], [3, 4, 5]]
test_y = [[1, 2, 3], [2, 3, 4], [3, 4, 5]]

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111)
ax.set_xlim([0, 5])
ax.set_ylim([0, 5])
points = ax.scatter(test_x[0], test_y[0], c='red')


def animate(i):
    ax.clf()
    points = ax.scatter(test_x[i], test_y[i], c='red')

    return points
    # use i-th elements from data


print(agents[0].pos[0])
print(agents[0].pos[1])

for i in range(20):
    for aj in agents:
        ai = find_nearest_neighbor(aj)
        aj.velo += w * aj.velo
        aj.velo = (aj.velo / np.linalg.norm(aj.velo))
        aj.pos += aj.velo

    x_list.append(get_x_pos())
    y_list.append(get_y_pos())


# ani = FuncAnimation(fig, animate, frames=20, interval=200)

def animate_test(i):
    ax.scatter(test_x[i], test_y[i])


ani2 = FuncAnimation(fig, animate_test, frames=3, interval=1000)

plt.show()