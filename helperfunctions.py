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

    # calculate angle between vectors
def exceeded_angle(vec_a, vec_b, max_angle=np.radians(40)):
    angle = np.arccos(np.dot(vec_a, vec_b)/(np.linalg.norm(vec_a)*np.linalg.norm(vec_b)))
    if np.abs(angle) > max_angle:
        return np.sign(angle - max_angle)
    return 0

def set_angle(vec_old, vec_new, norm, angle=np.radians(40)):
    exceeded = exceeded_angle(vec_old, vec_new, angle)
    if exceeded == 0:
        return vec_new
    angle *= -exceeded
    x, y = vec_old
    x_new = x*np.cos(angle) - y*np.sin(angle)
    y_new = x*np.sin(angle) + y*np.cos(angle)
    vec_new = np.array((x_new, y_new))
    vec_new = normalize(vec_new, norm)
    return vec_new