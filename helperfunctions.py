import numpy as np

def find_nearest_neighbor(aj, agents): 
    """Nearest neighbor for agents."""
    distances = [np.linalg.norm(np.array([aj.point.getCenter().getX(), aj.point.getCenter().getY()]) -
                                np.array([ai.point.getCenter().getX(), ai.point.getCenter().getY()])) for ai in agents]
    distances[distances.index(0.0)] = max(distances)
    return(agents[distances.index(min(distances))])

def find_k_nearest_neighbor(aj, k, agents): 
    """K-nearest neighbor function for agents."""
    distances = [np.linalg.norm(np.array([aj.point.getCenter().getX(), aj.point.getCenter().getY()]) -
                                np.array([ai.point.getCenter().getX(), ai.point.getCenter().getY()])) for ai in agents]
    sorted_distances = np.sort(distances)
    neighbours = []
    for x in sorted_distances[1:k+1]:
        neighbours.append(agents[distances.index(x)])
    return (neighbours)

def normalize(vec, norm):
    """Normalizes the vector and multiplies it with norm."""
    return(norm*vec/np.linalg.norm(vec))

def calculate_distances(aj, agents):
    """Calculate distances to all agents

    Returns distances from aj to all other agents and the vectors from aj to all other agents. 
    """
    pos_dif = [ np.array([aj.point.getCenter().getX(), aj.point.getCenter().getY()]) -
                np.array([ai.point.getCenter().getX(), ai.point.getCenter().getY()]) for ai in agents]
    dist = np.linalg.norm(np.array(pos_dif),axis=1)
    return([dist, pos_dif])

def calculate_distances_and_velo(aj, agents):
    """Calculate distances to all agents

    Returns distances from aj to all other agents and the vectors from aj to all other agents. 
    """
    values = [[np.linalg.norm(np.array([aj.point.getCenter().getX(), aj.point.getCenter().getY()]) -
                np.array([ai.point.getCenter().getX(), ai.point.getCenter().getY()])), ai.velo] for ai in agents]
    return(values)


# calculate angle between vectors
def angle_between(p1, p2):
    ang1 = np.arctan2(*p1[::-1])
    ang2 = np.arctan2(*p2[::-1])
    angle_360 = np.rad2deg((ang1 - ang2) % (2 * np.pi))
    if angle_360 <= 180:
        return angle_360
    else:
        return -1*(360 - angle_360)

def exceeded_angle(vec_old, vec_new, max_angle):
    angle = angle_between(vec_old, vec_new)
    if np.abs(angle) > max_angle:
        return np.sign(angle)
    return 0

def set_angle(vec_old, vec_new, norm, angle):
    exceeded = -1*exceeded_angle(vec_old, vec_new, angle)
    if exceeded == 0:
        return vec_new
    angle *= exceeded
    x, y = vec_old
    x_new = x*np.cos(angle) - y*np.sin(angle)
    y_new = x*np.sin(angle) + y*np.cos(angle)
    vec_new = np.array((x_new, y_new))
    vec_new = normalize(vec_new, norm)
    return vec_new
