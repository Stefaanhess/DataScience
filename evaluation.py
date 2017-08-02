import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

# evaluate std and mean 
def evaluate_current_timestep(agents):
    iu1 = np.triu_indices(len(agents),1)
    dist_m = distance_matrix(agents)
    return np.mean(dist_m[iu1]), np.std(dist_m[iu1])
    
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
    plt.savefig('evaluation')

def getCluster(agents):
    positions = np.array([ai.getPos() for ai in agents])
    cluster = DBSCAN(eps=40, min_samples=3).fit(positions)
    labels = cluster.labels_
    clusters = []
    no_cluster = []
    for label in set(labels):
        if label == -1:
            no_cluster = positions[(labels==label)]
        else:
            clusters.append(positions[(labels==label)])
    return no_cluster, clusters
    #sorted([(labels[i], pos) for i, pos  in enumerate(positions)], key=lambda x: x[0])
   # return positions, labels
#    unique_labels = set(labels)
#    colors = get_spaced_colors(len(unique_labels))
#    unique_labels = list(unique_labels)
#    for i, label in enumerate(labels):
#        index = unique_labels.index(label)
#        rgb = np.asarray(colors[index])
#        agents[i].setColor(color_rgb(rgb[0], rgb[1], rgb[2]))
#    print(colors)

if __name__=='__main__':
    a = np.array([False, True, False, True])
    b = np.array([[1, 1], [2, 2], [3, 3], [4, 4]])
    print(b[a])


