
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
minB = 15
maxB = winWidth-15

window = GraphWin("Window", winWidth, winHeight)
   
### Setting up the world    

def initialize():
    for i in range(N):
        agents.append(Agent(winWidth*np.random.random(2),8*np.random.random(2)-4))
    for ai in agents:
        ai.point.setFill("red")

# draw the agents into the window to display them
def draw_agents():
    for ai in agents:
        ai.point.draw(window)


### Simpel Update Function (Class 1)

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

### Algorithm of Couzin (Class 2)

### Behavourial Rules of Zoning

# global parameter for all agents

Rr = 10
Ro = 40
Ra = 80
winWidth = 1000

# calculate all distances to all agents
def calculate_distances(aj):
    pos_dif = [ np.array([aj.point.getCenter().getX(), aj.point.getCenter().getY()]) -
                np.array([ai.point.getCenter().getX(), ai.point.getCenter().getY()]) for ai in agents]
    dist = np.linalg.norm(np.array(pos_dif),axis=0)
    return([dist, pos_dif])


# check for agents in repulsion, orientation or attraction zone

def neighbours_in_zoneR(distances):
     for i in range(len(distances)):
        if(distances[i] > 0 and distances[i] < Rr):
            return True
        else:
            return False

def neighbours_in_zoneO(distances):
    for i in range(len(distances)):
        if(distances[i] >= Rr and distances[i] < Ro):
            return True
        else:
            return False
        
def neighbours_in_zoneA(distances):
    for i in range(len(distances)):
        if(distances[i] >= Ro and distances[i] < Ra):
            return True
        else:
            return False


# calculate new direction according to agents in zones

def calculate_direction_R(dist, pos_dif):
    direct = np.zeros(2)
    for i in range(len(dist)):
        if (dist[i] < Rr and dist[i] > 0):
            direct -= pos_dif[i]/dist[i]
    return direct

def calculate_direction_O(dist):
    direct = np.zeros(2)
    for i in range(len(dist)):
        if (dist[i] >= Rr and dist[i] < Ro):
            direct += agents[i].velo/np.linalg.norm(agents[i].velo)
    return direct

def calculate_direction_A(dist, pos_dif):
    direct = np.zeros(2)
    for i in range(len(dist)):
        if (dist[i] >= Ro and dist[i] < Ra):
            direct += pos_dif[i]/dist[i]
    return direct


# asses next step for agent

def asses_next_step(aj):
    dist = calculate_distances(aj)[0]
    pos_dif = calculate_distances(aj)[1]

    if (neighbours_in_zoneR(dist)):
        new_direct = calculate_direction_R(dist, pos_dif)
        
    elif (neighbours_in_zoneO(dist) and neighbours_in_zoneA(dist)):
        new_direct = 0.5*(calculate_direction_A(dist, pos_dif) + calculate_direction_O(dist))

    elif (neighbours_in_zoneO(dist)):
        new_direct = calculate_direction_O(dist)

    elif (neighbours_in_zoneA(dist)):
        new_direct = calculate_direction_A(dist, pos_dif)

    else:
        new_direct = np.zeros(2)

    aj.velo_temp = aj.velo + new_direct

# move the agent according to the velocity
def move_agent(aj):
    aj.point.move(aj.velo[0]+random.uniform(-3, 3),aj.velo[1]+random.uniform(-3, 3))

# first loop: calculate all new velos with old velos
# second loop: set value old velo to new velo and move agents
def update_agents():
    for aj in agents:
        # Algo 1
 #       assimilate_velo(aj);
        # Algo 2
        asses_next_step(aj)
    for ai in agents:
        ai.velo = ai.velo_temp
        move_agent(ai)

### Simulation

def do_simulation():
    for i in range(1000):
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
