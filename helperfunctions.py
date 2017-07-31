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

def add_noise(vec, noise=2):
    """
    add noise to the vector
    """
    if noise == 0:
        return vec
    norm = np.linalg.norm(vec)
    random_angle = (np.random.rand()-0.5)*2*noise
    random_angle = np.radians(random_angle)
    vec_new = turn_vector(vec, random_angle)
    vec_new = normalize(vec_new, norm)
    return vec_new


def angle_error(simulated, predicted, max_difference = 90):
    """
    calculate angle deviation between simulated and predicted velocities
    """
    angle_difference = angle_between(simulated, predicted)
    if np.abs(angle_difference) >= max_difference:
        (1, 0, 0)
    rel_diff = np.abs(angle_difference/max_difference)
    return (rel_diff, 1-rel_diff, 0)

def turn_vector(vec, angle):
    x, y = vec
    x_new = x*np.cos(angle) - y*np.sin(angle)
    y_new = x*np.sin(angle) + y*np.cos(angle)
    vec_new = np.array((x_new, y_new))
    return vec_new

def get_wall_intersection(position, direction, borders):
    s = np.inf
    # make numpy arrays
    if type(position) == list:
        position = np.array(position)
    if type(direction) == list:
        direction = np.array(direction)
    if type(borders) == list:
        borders = np.array(borders)

    for point in borders:
        s_tmp = (point[0] - position[0])/direction[0]
        if s_tmp < 0:
            s_tmp = np.inf
        s = min(s, s_tmp)
        s_tmp = (point[1] - position[1]) / direction[1]
        if s_tmp < 0:
            s_tmp = np.inf
        s = min(s, s_tmp)
    intersection_point = position + s * direction
    return intersection_point


if __name__ == "__main__":
    borders = [[0, 0], [100, 100]]
    position = [80, 80]
    directions = np.random.randint(-10, 10, size=(10, 2))
    for direction in directions:
        print('****')
        print(position, direction)
        print(get_wall_intersection(position, direction, borders))