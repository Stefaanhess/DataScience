import numpy as np

# evaluate std and mean 
def evaluate_current_timestep(agents):
    iu1 = np.triu_indices(len(agents),1)
    dist_m = distance_matrix(agents)
    mean_dist_list.append(np.mean(dist_m[iu1]))
    diviation_list.append(np.std(dist_m[iu1]))
    
# distance matrix
def distance_matrix(agents):
    distances = []
    for ai in agents:
        distances.append(np.linalg.norm(np.array(
            [np.array([aj.point.getCenter().getX(), aj.point.getCenter().getY()]) -
             np.array([ai.point.getCenter().getX(), ai.point.getCenter().getY()])
             for aj in agents]),axis=1))
    return(np.array(distances))

# plot evaluation graphs
def plot_graphs(mean_dist_list, diviation_list):
    f, axarr = plt.subplots(2, sharex=True)
    axarr[0].plot(mean_dist_list)
    axarr[0].set_ylabel('Mean Distance')
    axarr[1].plot(diviation_list)
    axarr[1].set_ylabel('Standard Deviation')
    plt.show()

