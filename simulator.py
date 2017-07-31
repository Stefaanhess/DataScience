
from graphics import *
from helperfunctions import *
from couzin import couzin_next_step
from simple_model import assimilate_velo
from vicsek import vicek_next_step
from tempfile import TemporaryFile
import colorsys
import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt

### CLASS AGENT

class Agent:
    def __init__(self, pos, velo, color):
        self.velo = velo
        self.velo_temp = velo
        self.point = Circle(Point(pos[0],pos[1]),3)
        self.point.setFill(color)
        self.history = []

    def getX(self): 
        return self.point.getCenter().getX()

    def getY(self):
        return self.point.getCenter().getY()

    def getPos(self):
        return np.array([self.getX(), self.getY()])

    def setColor(self,color):
        self.point.setFill(color)

    def appendTimestep(self, radius):
        array = np.array([self.getX(), self.getY(), self.velo[0], self.velo[1]])
        bins = get_agents_in_sight(self, agents, radius)
        wall_bins = get_walls_in_sight(self, radius=radius)
        self.history.append(np.concatenate([array, bins, wall_bins]))

    def __str__(self):
        return "velo = " + self.velo.__str__()  + "; position = (" + self.point.getCenter().getX().__str__() + ", " + self.point.getCenter().getY().__str__() +")"

### GLOBAL PARAMETERS
agents = []
N = 5
winWidth = 700
winHeight = 700
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

def digitize_track(track):
    discretization_bins = np.load("Bins.npy")
    for feature_i in range(4):
        track[:, feature_i] = np.digitize(track[:, feature_i], discretization_bins[feature_i], right=True)
    return track.astype(np.int32)
   
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


# first loop: calculate all new velos with old velos
# second loop: set value old velo to new velo and move agents
sess = tf.Session()
saver = tf.train.import_meta_graph('meta_graph_1')
saver.restore(sess, 'my-model_1')
def update_agents():
    for aj in agents:
        # Algo 1
        #assimilate_velo(aj, agents, minB, maxB);
        # Algo 2
        couzin_next_step(aj, agents, norm)
        # Algo 3
        # vicek_next_step
    for ai in agents:
        avoid_border_crossing(ai)
        ai.appendTimestep(200)
        if (len(ai.history)>50):
            track_continuous = np.expand_dims(np.array(ai.history[-50:]), axis=0)
            track_discrete = np.expand_dims(digitize_track(np.array(ai.history[-50:])), axis=0)
            print(track_continuous.shape)
            print(track_discrete.shape)
            #with tf.Session() as sess: 
                
            result = sess.run(['generator/gen_output:0'], feed_dict={'track_continuous:0': track_continuous, 'track_discrete:0': track_discrete})
            print(np.asarray(result).shape)
            print(result[0][-1])
            print((result[0][-1]).shape)

        ai.velo = ai.velo_temp
        move_agent(ai)

### Simulation
def do_simulation():
    for i in range(55):
        update_agents()
        evaluate_current_timestep()
    window.close()

def get_agents_in_sight(ai, agents, radius, num_bins = 36):
    """
    seperate environment of an agent into bins and find nearest neighbour for all bins.
    """
    bins = np.zeros(num_bins)
    borders = np.linspace(-180, 180, num_bins + 1)
    directions = [agent.getPos()-ai.getPos() for agent in agents if 0 < np.linalg.norm(agent.getPos()-ai.getPos()) <= radius]
    angles = [[angle_between(ai.velo, direction), direction] for direction in directions]
    for angle in angles:
        for i in range(len(borders)-1):
            if angle[0] <= borders[i+1]:
                bins[i] = max([bins[i], (radius - np.linalg.norm(angle[1]))/radius])
                break
    return bins

def get_walls_in_sight(ai, radius, num_bins = 72, borders=[[0,0], [winWidth, winHeight]]):
    """
    """
    walls = [[0, 0], borders]
    walls_dirs = [[0, 1], [1, 0]]
    bins = np.zeros(num_bins)
    position = ai.getPos()
    angles = np.linspace(-175, 175, num_bins, endpoint=True)
    angles = np.radians(angles)
    target_dirs = [normalize(turn_vector(ai.velo, angle), radius) for angle in angles]
    for i, direction in enumerate(target_dirs):
        intersection_point = get_wall_intersection(position, direction, borders)
        bins[i] = max(0, 1 - np.linalg.norm(position-intersection_point)/radius)
    return bins


### RUN IT
# plt.ion()

initialize()
draw_agents()
do_simulation()

#tracks = []
#for agent in agents:
#    tracks.append(agent.history)

#np_tracks = np.asarray(tracks)
#np.save("Tracks", np_tracks)
