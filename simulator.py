import numpy as np
import matplotlib.pyplot as plt
import threading,time
from . graphics import *

class Agent: 
    def __init__(self, pos, velo):
        self.velo = velo
        self.velo_temp = velo
        self.Point(pos[0],pos[1])

agents = []
w = 2
N = 100
winWidth = 1000
winHeight = 1000
   
### Updating the agents    

def initialize():
    for i in range(N):
        agents.append(Agent(winWidth*np.random.random(2),4*np.random.random(2)-2))

# draw the agents into the window to display them
def draw_agents():
    for ai in agents:
        ai.Point.draw(window)

def find_nearest_neighbor(aj): 
    distances = [np.linalg.norm(np.array([aj.Point.getX(), aj.Point.getY()]) -
                                np.array([ai.Point.getX(), ai.Point.getY()])) for ai in agents]
    distances[distances.index(0.0)] = max(distances)
    return(agents[distances.index(min(distances))])

# rename it to update velocity
def similar_direction_as_neighbour(aj):
    ai = find_nearest_neighbor(aj)
    aj.velo_temp = aj.velo + w*ai.velo
    aj.velo_temp = (aj.velo_temp / np.linalg.norm(aj.velo_temp))
    return 

def update_agents():
    for aj in agents:
        #similar_direction_as_neighbour(aj);
        go_towards_neighbour(aj)
    for ak in agents:
        ak.velo = ak.velo_temp
        move_agent(ak)
    return

# move the agent according to the velocity, do not exceed boundaries
def move_agent(aj):    
    x = aj.Point.getX()
    y = aj.Point.getY()

    if x > 0 and y < winHeight and x < winWidth and y > 0:
        aj.Point.move(aj.velo[0],aj.velo[1])

    elif x <= 0 or x >= winWidth:
        aj.Point.move(aj.velo[0]*(-1), aj.velo[1])

    elif y <= 0 or y >= winHeight:
        aj.Point.move(aj.velo[0], aj.velo[1]*(-1))

    return

### Simulation

def do_simulation():
    for i in range(100):
        update_agents()
        time.sleep(0.3)

    window.getMouse()
    window.close()    

window = GraphWin("Window", winWidth, winHeight)

initialize()
draw_agents()
do_simulation()
