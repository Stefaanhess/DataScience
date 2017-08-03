from helperfunctions import *

# weight parameter
w = 2

def assimilate_velo(aj, agents, minB, maxB):
    """Calculate the new velocity for a single agent by looking at the velocity of the nearest neighbour."""
    ai = find_nearest_neighbor(aj, agents)
    aj.velo_temp = aj.velo + w*ai.velo
    aj.velo_temp = (aj.velo_temp / np.linalg.norm(aj.velo_temp)*10)

    # avoiding boarder crossing
    if aj.point.getCenter().getX() <= minB or aj.point.getCenter().getX() >= maxB:
        aj.velo_temp[0] = aj.velo_temp[0]*(-1)
    elif aj.point.getCenter().getY() <= minB or aj.point.getCenter().getY() >= maxB:
        aj.velo_temp[1] = aj.velo_temp[1]*(-1)
