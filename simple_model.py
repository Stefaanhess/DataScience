from helperfunctions import *

# weight parameter
w = 0.5

def assimilate_velo(aj, agents, norm):
    """Calculate the new velocity for a single agent by looking at the velocity of the nearest neighbour."""
    ai = find_nearest_neighbor(aj, agents)
    aj.velo_temp = aj.velo + w*ai.velo
    aj.velo_temp = (aj.velo_temp / np.linalg.norm(aj.velo_temp)*norm)

