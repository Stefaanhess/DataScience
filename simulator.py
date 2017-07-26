
from graphics import *
from helperfunctions import *
from couzin import couzin_next_step
from simple_model import assimilate_velo

import numpy as np
import time
import random
import math

### CLASS AGENT

class Agent:
    def __init__(self, pos, velo):
        self.velo = velo
        self.velo_temp = velo
        self.point = Circle(Point(pos[0],pos[1]),3)

### GLOBAL PARAMETERS

agents = []
N = 40
winWidth = 1000
winHeight = 1000

# turning points at border
minB = 15
maxB = winWidth-15


# global parameter for all agents, all radius and speed
norm = 6

window = GraphWin("Window", winWidth, winHeight)
   
### SETTING UP THE WORLD    

def initialize():
    for i in range(N):
        agents.append(Agent(winWidth/2*np.random.random(2)+winWidth*(1/4),normalize(4*np.random.random(2)-2, norm)))
    for ai in agents:
        ai.point.setFill("red")

# draw the agents into the window to display them
def draw_agents():
    for ai in agents:
        ai.point.draw(window)

# function to avoid border crossing
def avoid_border_crossing(aj):
    if aj.point.getCenter().getX() <= minB or aj.point.getCenter().getX() >= maxB:
        aj.velo_temp[0] = aj.velo_temp[0]*(-1)
    elif aj.point.getCenter().getY() <= minB or aj.point.getCenter().getY() >= maxB:
        aj.velo_temp[1] = aj.velo_temp[1]*(-1)


# move the agent according to the velocity
def move_agent(aj):
    aj.point.move(aj.velo[0],aj.velo[1])

# first loop: calculate all new velos with old velos
# second loop: set value old velo to new velo and move agents
def update_agents():
    for aj in agents:
        # Algo 1
        #assimilate_velo(aj, agents, minB, maxB);
        # Algo 2
        couzin_next_step(aj, agents, norm)
    for ai in agents:
        avoid_border_crossing(ai)
        ai.velo = ai.velo_temp
        move_agent(ai)

### Simulation
def do_simulation():
    for i in range(3000):
        update_agents()
    window.getMouse()

### RUN IT

initialize()
draw_agents()
do_simulation()