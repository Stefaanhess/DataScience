from helperfunctions import *

r = 20
ref_vector = [1,0]
eta = 20

def vicek_next_step(aj, agents, norm):
    """Asses the next step for a single agent according to the Vicsek paper."""
    avrg_velo = get_average_velos_in_neighbourhood(aj,agents)
    avrg_angle = angle_between(ref_vector, avrg_velo)
    avrg_angle = add_noise(avrg_angle)
    new_velo = get_vector_from_angle(norm, avrg_angle)
    aj.velo_temp = normalize(new_velo, norm)

def get_average_velos_in_neighbourhood(aj, agents):
    dist_velo_pairs = calculate_distances_and_velo(aj, agents) 
    total_velo = np.array([0.0,0.0])
    for pair in dist_velo_pairs:
    	if pair[0] < r:
    	    total_velo += pair[1]
    return total_velo

def add_noise(angle):
	return (np.random.rand()-0.5)*eta + angle

def get_vector_from_angle(norm, angle):
    x, y = ref_vector
    angle = - np.radians(angle)
    x_new = x*np.cos(angle) - y*np.sin(angle)
    y_new = x*np.sin(angle) + y*np.cos(angle)
    vec_new = np.array((x_new, y_new))
    vec_new = normalize(vec_new, norm)
    return vec_new

