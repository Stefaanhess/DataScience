
from graphics import *
from helperfunctions import *
from couzin import couzin_next_step
from simple_model import assimilate_velo
from vicsek import vicek_next_step
from tempfile import TemporaryFile
import colorsys

import numpy as np
import matplotlib.pyplot as plt

### CLASS AGENT

class Agent:
    def __init__(self, pos, velo, color):
        self.velo = velo
        self.velo_temp = velo
        self.point = Circle(Point(pos[0],pos[1]),3)
        self.point.setFill(color)

    def getX(self): 
        return self.point.getCenter().getX()

    def getY(self):
        return self.point.getCenter().getY()

    def getPos(self):
        return np.array([self.getX(), self.getY()])

    def setColor(self,color):
        self.point.setFill(color)

    def __str__(self):
        return "velo = " + self.velo.__str__()  + "; position = (" + self.point.getCenter().getX().__str__() + ", " + self.point.getCenter().getY().__str__() +")"

### GLOBAL PARAMETERS
tracks = np.array([])
agents = []
N = 20
winWidth = 400
winHeight = 400
window = GraphWin("Window", winWidth, winHeight)

# evaluation parameter
mean_dist_list = []
diviation_list = []

# global parameter for all agents, all radius and speed
norm = 5

### Evaluation Functions

# evaluate std and mean 
def evaluate_current_timestep():
    iu1 = np.triu_indices(len(agents),1)
    dist_m = distance_matrix()
    mean_dist_list.append(np.mean(dist_m[iu1]))
    diviation_list.append(np.std(dist_m[iu1]))
    
# distance matrix
def distance_matrix():
    distances = []
    for ai in agents:
        distances.append(np.linalg.norm(np.array(
            [np.array([aj.point.getCenter().getX(), aj.point.getCenter().getY()]) -
             np.array([ai.point.getCenter().getX(), ai.point.getCenter().getY()])
             for aj in agents]),axis=1))
    return(np.array(distances))

# plot the graphs
def plot_graphs():
    f, axarr = plt.subplots(2, sharex=True)
    axarr[0].plot(mean_dist_list)
    axarr[0].set_ylabel('Mean Distance')
    axarr[1].plot(diviation_list)
    axarr[1].set_ylabel('Standard Deviation')
    plt.show()
   
### SETTING UP THE WORLD    

def initialize():
    for i in range(N):
        agents.append(Agent(winWidth*np.random.random(2),normalize(4*np.random.random(2)-2, norm),'red'))

# draw the agents into the window to display them
def draw_agents():
    for ai in agents:
        ai.point.draw(window)

# function to avoid border crossing
def avoid_border_crossing(aj):
    if aj.point.getCenter().getX() + aj.velo_temp[0] <= 0 or aj.point.getCenter().getX() + aj.velo_temp[0] >= winWidth:
        aj.velo_temp[0] = aj.velo_temp[0]*(-1)
    if aj.point.getCenter().getY() + aj.velo_temp[1] <= 0 or aj.point.getCenter().getY() + aj.velo_temp[1] >= winHeight:
        aj.velo_temp[1] = aj.velo_temp[1]*(-1)


def allow_border_crossing(aj):
    if aj.point.getCenter().getX() <= 0:
        aj.point.move(winWidth,0)
    elif  aj.point.getCenter().getX() >= winWidth:
        aj.point.move(-winWidth,0)
    elif  aj.point.getCenter().getY() <= 0:
        aj.point.move(0,winHeight)
    elif  aj.point.getCenter().getY() >= winHeight:
        aj.point.move(0,-winHeight)

# move the agent according to the velocity
def move_agent(aj):
    aj.point.move(aj.velo[0],aj.velo[1])

# TODO update color according to error in prediction
def update_color(aj):
    # c = angle_error() # needs real value and estimated one!
    t = colorsys.hsv_to_rgb((c*0.33),0.8,0.8)
    t = np.array(t)*255
    aj.setColor(color_rgb(int(t[0]),int(t[1]),int(t[2])))
    
track_all = []
# first loop: calculate all new velos with old velos
# second loop: set value old velo to new velo and move agents
def update_agents():
    track_t = []
    for aj in agents:
        # Algo 1
        #assimilate_velo(aj, agents, minB, maxB);
        # Algo 2
        couzin_next_step(aj, agents, norm)
        # Algo 3
        # vicek_next_step
    for ai in agents:
        allow_border_crossing(ai)
        track_t.append(np.array([ai.point.getCenter().getX(), ai.point.getCenter().getY(), angle_between([1,0], ai.velo)]))
        ai.velo = ai.velo_temp
        move_agent(ai)
    track_all.append(track_t)

### Simulation
def do_simulation():
    for i in range(1000):
        update_agents()
        evaluate_current_timestep()
    window.close()

### RUN IT

initialize()
draw_agents()
do_simulation()

np_tracks = np.asarray(track_all)
np_tracks = np.swapaxes(np_tracks,0,1)
np.save("Tracks", np_tracks)
