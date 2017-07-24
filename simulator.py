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
w = 0.75
N = 20
box_size = 10
   
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
    pos_temp = aj.pos + aj.velo_temp
    if is_outside_box(pos_temp):
        print(aj.pos)
        print(aj.velo_temp)
        aj.velo_temp = invert_velocity(aj)
        print(aj.velo_temp)
    aj.pos += aj.velo_temp   
    return

def update_agents():
    for aj in agents:
        #similar_direction_as_neighbour(aj);
        go_towards_neighbour(aj)
    for ak in agents:
        ak.velo = ak.velo_temp
    return

def invert_velocity(agent):
    pos = agent.pos
    velo = agent.velo_temp
    if np.abs(pos[0]) < np.abs(pos[1]):
        inv_index = 1
    else:
        inv_index = 0
    velo[inv_index] *= -1
    return velo

def is_outside_box(pos):
    if np.abs(pos[0]) > box_size or np.abs(pos[1]) > box_size:
        return True
    return False

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
    ax.set_xlim([-box_size, box_size])
    ax.set_ylim([-box_size, box_size])
    return 


### Simulation

initialize()

fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111)
ax.set_xlim([-box_size, box_size])
ax.set_ylim([-box_size, box_size])

def do_simulation():
    for i in range(100):
        update_agents()
        draw_simulation()
        time.sleep(0.1)

t = threading.Thread(target=do_simulation)
t.start()
plt.show()
