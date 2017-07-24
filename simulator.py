import numpy as np
import matplotlib.pyplot as plt
import threading,time

class Agent: 
    def __init__(self, pos, velo):
        self.pos = pos
        self.velo = velo
        self.velo_temp = velo

x_list = []
y_list =[]
agents = []
w = 0.5
N = 4
   
### Updating the agents    

def initialize():
    for i in range(N):
        agents.append(Agent(10*np.random.random(2)-5,2*np.random.random(2)-1))

'''
def find_nearest_neighbor(aj): 
    distances = [np.linalg.norm(aj.velo - ai.velo) for ai in agents]
    distances[distances.index(0.0)] = max(distances)
    return(agents[distances.index(min(distances))])
'''


def find_nearest_neighbor(aj):
    distances = [np.linalg.norm(aj.pos - ai.pos) for ai in agents]
    distances[distances.index(0.0)] = max(distances)
    return (agents[distances.index(min(distances[1:]))]) # ignores the distance to itself which is always 0

def similar_direction_as_neighbour(aj):
    ai = find_nearest_neighbor(aj)
    aj.velo_temp = aj.velo + w*ai.velo
    aj.velo_temp = (aj.velo_temp / np.linalg.norm(aj.velo_temp))
    aj.pos += aj.velo_temp    
    return 

def go_towards_neighbour(aj):
    ai = find_nearest_neighbor(aj)
    aj.velo_temp = aj.velo + w * np.array([ai.pos[0] - aj.pos[0], ai.pos[1] - ai.pos[1]])
    aj.velo_temp = (aj.velo_temp / np.linalg.norm(aj.velo_temp))
    aj.pos += aj.velo_temp   
    return

def update_agents():
    for aj in agents:
        #similar_direction_as_neighbour(aj);
        go_towards_neighbour(aj)
    for ak in agents:
        ak.velo = ak.velo_temp
    return

### Plotting the simulation

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

def draw_simulation():
    ax.cla()
    ax.scatter(get_x_pos(), get_y_pos())
    plt.draw()
    ax.set_xlim([-20, 20])
    ax.set_ylim([-20, 20])
    return 

### Simulation

initialize()

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111)
ax.set_xlim([-20, 20])
ax.set_ylim([-20, 20])

def do_simulation():
    for i in range(100):
        update_agents()
        draw_simulation()
        time.sleep(1)


t = threading.Thread(target=do_simulation)
t.start()
plt.show()
