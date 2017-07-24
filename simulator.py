
from graphics import *

import numpy as np
import time
import random
import math

class Agent:
    def __init__(self, pos, velo):
        self.velo = velo
        self.velo_temp = velo
        self.point = Circle(Point(pos[0],pos[1]),3)

agents = []
w = 2
N = 40
winWidth = 1000
winHeight = 1000
window = GraphWin("Window", winWidth, winHeight)
   
### Setting up the world    

def initialize():
    for i in range(N):
        agents.append(Agent(winWidth*np.random.random(2),4*np.random.random(2)-2))
    for ai in agents:
        ai.point.setFill("red")

# draw the agents into the window to display them
def draw_agents():
    for ai in agents:
        ai.point.draw(window)


### Updating Agents

# nearest neighbor funtion
def find_nearest_neighbuor(aj): 
    distances = [np.linalg.norm(np.array([aj.point.getCenter().getX(), aj.point.getCenter().getY()]) -
                                np.array([ai.point.getCenter().getX(), ai.point.getCenter().getY()])) for ai in agents]
    distances[distances.index(0.0)] = max(distances)
    return(agents[distances.index(min(distances))])

# calculate the new velo including velo of nearest neighbour
def assimilate_velo(aj):
    ai = find_nearest_neighbuor(aj)
    aj.velo_temp = aj.velo + w*ai.velo
    aj.velo_temp = (aj.velo_temp / np.linalg.norm(aj.velo_temp)*10)

    # avoiding boarder crossing
    if aj.point.getCenter().getX() <= 0 or aj.point.getCenter().getX() >= winWidth:
        aj.velo_temp[0] = aj.velo_temp[0]*(-1)
    elif aj.point.getCenter().getY() <= 0 or aj.point.getCenter().getY() >= winHeight:
        aj.velo_temp[1] = aj.velo_temp[1]*(-1)

# move the agent according to the velocity
def move_agent(aj):
    aj.point.move(aj.velo[0]+random.uniform(-3, 3),aj.velo[1]+random.uniform(-3, 3))

# first loop: calculate all new velos with old velos
# second loop: set value old velo to new velo and move agents
def update_agents():
    for aj in agents:
        assimilate_velo(aj);
    for ai in agents:
        ai.velo = ai.velo_temp
        move_agent(ai)

### Simulation

def do_simulation():
    for i in range(100):
        update_agents()
 #       time.sleep(0.01)

    window.getMouse()

### RUN IT
    
initialize()
draw_agents()
do_simulation()

### Some Methods names
### Point exchanged with Circle to better illustrate
### Random Noise

### NOW: How to avoid all sticking to the borders?
