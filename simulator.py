from graphics import *
from helperfunctions import *
from couzin import couzin_next_step
from simple_model import assimilate_velo
from vicsek import vicek_next_step
from tempfile import TemporaryFile
if __name__ == '__main__':
    from network_model import *
from evaluation import *
from Settings import Settings

import colorsys
import tensorflow as tf

### GLOBAL PARAMETERS
settings = Settings()

T = 5000
N = settings.N
winWidth = settings.winWidth
winHeight = settings.winHeight
vision_radius = settings.vision_radius
vision_bins = settings.vision_bins
norm = settings.norm

position_history = []
velocity_history = []
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

    def appendTimestep(self, radius, vision_bins):
        array = np.array([self.velo[0], self.velo[1]])
        bins = get_agents_in_sight(self, agents, radius, vision_bins)
        wall_bins = get_walls_in_sight(self, radius, vision_bins, [[0,0], [winWidth, winHeight]] )
        self.history.append(np.concatenate([array, bins, wall_bins]))

    def __str__(self):
        return "velo = " + self.velo.__str__()  + "; position = (" + self.point.getCenter().getX().__str__() + ", " + self.point.getCenter().getY().__str__() +")"

### GLOBAL PARAMETERS
agents = []

### SETTING UP THE WORLD    

def initialize(num_agents, draw_active, window):
    global agents
    agents = []
    rand_width = winWidth * np.sort(np.random.rand(2))
    rand_height = winHeight * np.sort(np.random.rand(2))
    for i in range(num_agents):
        agents.append(Agent(np.array([np.random.uniform(rand_width[0], rand_width[1]), np.random.uniform(rand_height[0], rand_height[1])]),normalize(4*np.random.random(2)-2, norm),'blue'))
    if draw_active:
        draw_agents(window)

# draw the agents into the window to display them
def draw_agents(window):
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

means = []
stds = []

def update_agents(algorithm):
    for aj in agents:
        # Algo 1
        if algorithm == 1:
            assimilate_velo(aj, agents, norm)
        # Algo 2
        if algorithm == 2:
            couzin_next_step(aj, agents, norm)
        # Algo 3
        if algorithm == 3:
            vicek_next_step(aj, agents, norm)
        # Algo 4
        if algorithm == 4:
            run_network(aj, agents, norm)
            aj.velo_temp = normalize(aj.velo_temp, norm)
    for ai in agents:
        avoid_border_crossing(ai)
        ai.appendTimestep(vision_radius, vision_bins)
        ai.velo = ai.velo_temp        
        move_agent(ai)


### Simulation
def do_simulation(num_agents, num_timesteps, draw_active, algorithm=2, save_positions=False, show_vision=False):
    if draw_active:
        window = GraphWin("Window", winWidth, winHeight, autoflush=False)
        initialize(num_agents, draw_active, window)
    else:
        initialize(num_agents, draw_active, None)
    for i in range(num_timesteps):
        print("Timesteps: ", i)
        update_agents(algorithm)
        if draw_active:
            window.update()
        if save_positions:
            positions = [agent.getPos() for agent in agents]
            velocities = [agent.velo for agent in agents]
            position_history.append(positions)
            velocity_history.append(velocities)
        if show_vision:
            agents[0].setColor('red')
            others, wall = vision_of_agent(agents, 200, vision_bins)
            x = np.arange(len(others))
            ax1.cla()
            ax2.cla()
            ax1.bar(x, others)
            ax2.bar(x, wall)
            plt.draw()
    if draw_active:
        window.close()
    if save_positions:
        np.save("evaluation/rnn_positions", position_history)
        np.save("evaluation/rnn_velocities", velocity_history)
    return agents

def save_simulation(num_agents, num_timesteps, algorithm=2):
    initialize(num_agents, False, None)
    for i in range(num_timesteps):
        update_agents(algorithm)
    return agents
### RUN IT
# plt.ion()

if __name__=='__main__':
    #plt.ion()
    #f, axarr = plt.subplots(2, sharex=True)
    #ax1, ax2 = axarr
    do_simulation(N, T, True, 4, False, False)
    #plot_graphs(means, stds)

