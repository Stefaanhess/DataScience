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

def getCluster(positions, clustersize=3):
    #positions = np.array([ai.getPos() for ai in agents])
    cluster = DBSCAN(eps=40, min_samples=clustersize).fit(positions)
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

def cluster_density(positions):
    mean = np.mean(positions)
    max_dist = max([np.linalg.norm(mean - position) for position in positions])
    area = np.pi * max_dist**2
    density = len(positions) / area
    return density

def average_cluster_density(clusters):
    av_density = 0
    sum_cluster_size = 0
    for cluster in clusters:
        cluster_density = cluster_density(cluster)
        cluster_size = len(cluster)
        sum_cluster_size += cluster_size
        av_density += cluster_density * cluster_size
    av_density /= sum_cluster_size
    return av_density

def cluster_plot(data):
    f, axarr = plt.subplots(2, sharex=True)
    time, n_clusters, not_clustered = data
    ax1, ax2 = axarr
    ax1.set_title('number of clusters')
    ax1.plot(time, n_clusters, label='clusters >= 2')
    ax2.set_title('particles not clustered')
    ax2.plot(time, not_clustered)
    ax2.set_xlabel("timesteps")
    plt.savefig("figures/rnn_clusters.pdf")
    plt.show()









if __name__=='__main__':
    position_history = np.load("Evaluation/rnn_positions.npy")
    timestep = []
    n_clusters = []
    n_not_clustered = []
    for i, positions in enumerate(position_history):
        not_clustered, clusters = getCluster(positions)
        timestep.append(i)
        n_clusters.append(len(clusters))
        n_not_clustered.append(len(not_clustered))
    cluster_data = [timestep, n_clusters, n_not_clustered]
    cluster_plot(cluster_data)




