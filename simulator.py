
from graphics import *

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

window = GraphWin("Window", winWidth, winHeight)
   
### SETTING UP THE WORLD    

def initialize():
    for i in range(N):
        agents.append(Agent(winWidth/2*np.random.random(2)+winWidth*(1/4),normalize(4*np.random.random(2)-2)))
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

# normalizing function to define speed of agents
def normalize(vec):
    return(norm*vec/np.linalg.norm(vec))




### SIMPEL FUNCTION DAY 1 
### MOUDLE 1

# weight parameter
w = 2

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
    if aj.point.getCenter().getX() <= minB or aj.point.getCenter().getX() >= maxB:
        aj.velo_temp[0] = aj.velo_temp[0]*(-1)
    elif aj.point.getCenter().getY() <= minB or aj.point.getCenter().getY() >= maxB:
        aj.velo_temp[1] = aj.velo_temp[1]*(-1)

### ALGORITHM OF COUZIN - CLASS 2
### MODULE 2

# global parameter for all agents, all radius and speed

Rr = 20
Ro = 60
Ra = 90
norm = 5

# calculate all distances to all agents
def calculate_distances(aj):
    pos_dif = [ np.array([aj.point.getCenter().getX(), aj.point.getCenter().getY()]) -
                np.array([ai.point.getCenter().getX(), ai.point.getCenter().getY()]) for ai in agents]
    dist = np.linalg.norm(np.array(pos_dif),axis=1)
    return([dist, pos_dif])


# check for agents in repulsion, orientation or attraction zone

def neighbours_in_zoneR(distances):
    for i in range(len(distances)):
        if(distances[i] > 0.0 and distances[i] < Rr):
            return(True)
    return(False)

def neighbours_in_zoneO(distances):
    for i in range(len(distances)):
        if(distances[i] >= Rr and distances[i] < Ro):
            return(True)
    return(False)
        
def neighbours_in_zoneA(distances):
    for i in range(len(distances)):
        if(distances[i] >= Ro and distances[i] < Ra):
            return(True)
    return(False)


# calculate new direction according to agents in zones

def calculate_direction_R(dist, pos_dif):
    direct = np.zeros(2)
    for i in range(len(dist)):
        if (dist[i] < Rr and dist[i] > 0.0):
            direct = direct + normalize((pos_dif[i]/dist[i]))
    return direct

def calculate_direction_O(dist):
    direct = np.zeros(2)
    for i in range(len(dist)):
        if (dist[i] >= Rr and dist[i] < Ro):
            direct += normalize(agents[i].velo/np.linalg.norm(agents[i].velo))
    return direct

def calculate_direction_A(dist, pos_dif):
    direct = np.zeros(2)
    for i in range(len(dist)):
        if (dist[i] >= Ro and dist[i] < Ra):
            direct = direct - normalize((pos_dif[i]/dist[i]))
    return direct

# asses next step for agent

def asses_next_step(aj):
    dist = calculate_distances(aj)[0]
    pos_dif = calculate_distances(aj)[1]

    if (neighbours_in_zoneR(dist)==True):
        new_direct = calculate_direction_R(dist, pos_dif)
        
    elif (neighbours_in_zoneO(dist)==True and neighbours_in_zoneA(dist)==True):
        new_direct = 0.5*(calculate_direction_A(dist, pos_dif) + calculate_direction_O(dist))

    elif (neighbours_in_zoneO(dist)==True):
        new_direct = calculate_direction_O(dist)

    elif (neighbours_in_zoneA(dist)==True):
        new_direct = calculate_direction_A(dist, pos_dif)

    else:
        new_direct = np.zeros(2)

    aj.velo_temp = normalize(aj.velo + new_direct)


### END OF MODULE 2


# move the agent according to the velocity
def move_agent(aj):
    aj.point.move(aj.velo[0],aj.velo[1])

# first loop: calculate all new velos with old velos
# second loop: set value old velo to new velo and move agents
def update_agents():
    for aj in agents:
        # Algo 1
 #       assimilate_velo(aj);
        # Algo 2
        asses_next_step(aj)
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
