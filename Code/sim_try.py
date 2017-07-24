import numpy as np
import matplotlib.pyplot as plt
import threading,time

class Agent:
    def __init__(self, pos, velo):
        self.pos = pos
        self.velo_temp = velo
        self.velo = velo

x_list = []
y_list =[]
agents = []
w = 0.5
N = 5

def initialize():

    for i in range(N):
        agents.append(Agent(2*np.random.random(2),2*np.random.random(2)))

def find_nearest_neighbor(aj):
    distances = [np.linalg.norm(aj.velo - ai.velo) for ai in agents]
    distances[distances.index(0.0)] = max(distances)
    return(agents[distances.index(min(distances))])

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

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111)
ax.set_xlim([0, 20])
ax.set_ylim([0, 20])

print(agents[0].pos[0])
print(agents[0].pos[1])

def do_simulation():
    for i in range(20):
        for aj in agents:
            ai = find_nearest_neighbor(aj)
            aj.velo += w*aj.velo
            aj.velo = (aj.velo / np.linalg.norm(aj.velo))
            aj.pos += aj.velo

        time.sleep(1)
        ax.cla()
        ax.scatter(get_x_pos(), get_y_pos())
        plt.draw()

t = threading.Thread(target=do_simulation)
t.start()
plt.show()