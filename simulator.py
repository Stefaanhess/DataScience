
from graphics import *

import numpy as np
import time
import random
import math

class Agent:
    def __init__(self, pos, velo):
        self.velo = velo
        self.velo_temp = velo
        self.point = Point(pos[0],pos[1])

agents = []
w = 2
N = 40
winWidth = 1000
winHeight = 1000
window = GraphWin("Window", winWidth, winHeight)
   
### Updating the agents    

def initialize():
    for i in range(N):
        agents.append(Agent(winWidth*np.random.random(2),4*np.random.random(2)-2))

# draw the agents into the window to display them
def draw_agents():
    for ai in agents:
        ai.point.draw(window)

def find_nearest_neighbor(aj): 
    distances = [np.linalg.norm(np.array([aj.point.getX(), aj.point.getY()]) -
                                np.array([ai.point.getX(), ai.point.getY()])) for ai in agents]
    distances[distances.index(0.0)] = max(distances)
    return(agents[distances.index(min(distances))])

# rename it to update velocity
def similar_direction_as_neighbour(aj):
    ai = find_nearest_neighbor(aj)
    aj.velo_temp = aj.velo + w*ai.velo
    aj.velo_temp = (aj.velo_temp / np.linalg.norm(aj.velo_temp)*10)
    return 

def update_agents():
    for aj in agents:
        similar_direction_as_neighbour(aj);
    for ak in agents:
        ak.velo = ak.velo_temp
        ak = move_agent(ak)

# move the agent according to the velocity, do not exceed boundaries
def move_agent(aj):    
    x = aj.point.getX()
    y = aj.point.getY()

    if x > 0 and y < winHeight and x < winWidth and y > 0:
        aj.point.move(aj.velo[0],aj.velo[1])

    elif x <= 0 or x >= winWidth:
        aj.point.move(aj.velo[0]*(-1), aj.velo[1])

    elif y <= 0 or y >= winHeight:
        aj.point.move(aj.velo[0], aj.velo[1]*(-1))

### Simulation

def do_simulation():
    for i in range(100):
        update_agents()
 #       time.sleep(0.01)

    window.getMouse()
    window.close()    

initialize()
draw_agents()
do_simulation()
