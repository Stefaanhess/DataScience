import numpy as np
from matplotlib import pyplot as plt

plt.ion()

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
        return 1 
    rel_diff = np.abs(angle_difference/max_difference)
    return rel_diff

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


def get_bin_means(bin_borders):
    if type(bin_borders) == list:
        bin_borders = np.array(bin_borders)
    if len(bin_borders) <= 1:
        return False
    diff = np.abs(bin_borders[1] - bin_borders[0])

    bin_borders = np.array([border - diff/2 for border in bin_borders])
    bin_borders = np.append(bin_borders, bin_borders[-1] + diff)
    return bin_borders

plt.ion()
def plot_vision(others, wall, axarr):
    x = np.arange(len(others))
    ax1, ax2 = axarr
    ax1.bar(x, others)
    ax2.bar(x, wall)
    plt.draw()

def vision_of_agent(agents, radius, num_bins=36):
    agent_id = 0
    ai = agents[agent_id]
    other_agents = get_agents_in_sight(ai, agents, radius, num_bins)
    walls = get_walls_in_sight(ai, radius, num_bins)
    return other_agents, walls


def get_agents_in_sight(ai, agents, radius, num_bins = 36):
    """
    Seperate environment of an agent into bins and find nearest neighbour for all bins.
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

def get_walls_in_sight(ai, radius, num_bins = 36, borders=[[0,0], [700, 700]]):
    """
    Ray tracing in order to detect walls.
    """
    bins = np.zeros(num_bins)
    position = ai.getPos()
    angles = np.linspace(-175, 175, num_bins, endpoint=True)
    angles = np.radians(angles)
    target_dirs = [normalize(turn_vector(ai.velo, angle), radius) for angle in angles]
    for i, direction in enumerate(target_dirs):
        intersection_point = get_wall_intersection(position, direction, borders)
        bins[i] = max(0, 1 - np.linalg.norm(position-intersection_point)/radius)
    return bins

# TODO update color according to error in prediction
def update_color(aj, c):
    c = 1-c
    t = colorsys.hsv_to_rgb((c*0.33),0.8,0.8)
    t = np.array(t)*255
    aj.setColor(color_rgb(int(t[0]),int(t[1]),int(t[2])))


def merge_data_sets(files):
    arrays = [np.load(file) for file in files]
    arrays = tuple(arrays)
    merged_array = np.concatenate(arrays)
    np.save("Tracks/merged_tracks", merged_array)

def merge_tracks_to_file(merge_file, tracks):
    arrays = [np.load(merge_file), tracks]
    arrays = tuple(arrays)
    merged_array = np.concatenate(arrays)
    np.save(merge_file, merged_array)

def get_spaced_colors(n):
    """Get n distinct RGB values"""
    max_value = 16581375 #255**3
    interval = int(max_value / n)
    colors = [hex(I)[2:].zfill(6) for I in range(0, max_value, interval)]
    return [(int(i[:2], 16), int(i[2:4], 16), int(i[4:], 16)) for i in colors]

if __name__ == "__main__":
    #merge_data_sets(["Tracks.npy", "Tracks2.npy"])
    plt.ion()
    f, axarr = plt.subplots(2)
    ax1, ax2 = axarr
    for i in range(100):
        x = range(i)
        y = range(i)
        # plt.gca().cla() # optionally clear axes
        ax1.plot(x, y)
        ax2.plot(x, y)
        plt.draw()
        plt.pause(0.1)
