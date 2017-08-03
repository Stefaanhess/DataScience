import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from helperfunctions import *
import matplotlib
matplotlib.rcParams.update({'font.size': 18})


# evaluate std and mean 
def evaluate_current_timestep(agents):
    iu1 = np.triu_indices(len(agents),1)
    dist_m = distance_matrix(agents)
    return np.mean(dist_m[iu1]), np.std(dist_m[iu1])
    
# distance matrix
def distance_mean_std(positions):
    distances = []
    for i in range(len(positions)):
        for j in range(i, len(positions)):
            distances.append(np.linalg.norm(positions[i] - positions[j]))
    return np.mean(distances), np.std(distances)

# plot evaluation graphs
def distance_plot(data, outfile):
    timesteps, means, stds = data
    f, axarr = plt.subplots(2, sharex=True, figsize=(8, 8))
    ax1, ax2 = axarr
    ax1.plot(timesteps, means, label="mean")
    ax1.legend(loc=2)
    ax1.set_ylabel('Pixels')
    ax1.set_title("Average Particle Distance")
    ax1.set_ylim([0, max(means)*1.1])
    ax2.plot(timesteps, stds, label="std")
    ax2.legend(loc=2)
    ax2.set_ylabel('Pixels')
    ax2.set_xlabel('Timesteps')
    ax2.set_ylim([0, max(stds) * 1.1])
    plt.savefig(outfile)

clustersize = 4
def getCluster(positions):
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
    if len(clusters) == 0:
        return 0
    for cluster in clusters:
        cluster_size = len(cluster)
        sum_cluster_size += cluster_size
        av_density += cluster_density(cluster) * cluster_size
    av_density /= sum_cluster_size
    return 1/av_density

def cluster_plot(data, outfile):
    f, axarr = plt.subplots(2, sharex=True, figsize=(8, 8))
    time, n_clusters, not_clustered = data
    ax1, ax2 = axarr
    ax1.set_title('Number of Clusters')
    ax1.set_ylabel('N')
    ax1.plot(time, n_clusters, label='min clustersize = {}'.format(clustersize))
    ax1.legend(loc=2)
    ax2.set_title('Particles not Clustered')
    ax2.plot(time, not_clustered)
    ax2.set_ylabel('N')
    ax2.set_xlabel('Timesteps')
    plt.savefig(outfile)

def polarization_plot(data, outfile):
    f, axarr = plt.subplots(2, sharex=True, figsize=(8, 8))
    time, mean, std = data
    ax1, ax2 = axarr
    ax1.set_ylim([-180, 180])
    ax1.set_title('Polarization')
    ax1.plot(time, mean, label="mean")
    ax1.legend(loc=2)
    ax2.set_ylim([0, max(std)*1.1])
    ax2.plot(time, std, label="std")
    ax2.legend(loc=2)
    ax2.set_xlabel('Timesteps')
    plt.savefig(outfile)

def clusterdensity_plot(data, outfile):
    f, ax1 = plt.subplots(1, figsize=(8, 8))
    time, cluster_density = data
    ax1.set_title('Area per Particle')
    ax1.set_ylabel("Area")
    ax1.set_xlabel("Timesteps")
    ax1.set_ylim([0, max(cluster_density)*1.1])
    ax1.plot(time, cluster_density, label='min clustersize = {}'.format(clustersize))
    ax1.legend(loc=2)
    plt.savefig(outfile)

def get_polarization(velocities):
    angles = [angle_between([1, 0], vel) for vel in velocities]
    return np.mean(angles), np.std(angles)

def evaluate_positions(infile, density_outfile, cluster_outfile, distance_outfile):
    position_history = np.load(infile)
    timestep = []
    n_clusters = []
    n_not_clustered = []
    densities = []
    means = []
    stds = []

    for i, positions in enumerate(position_history):
        not_clustered, clusters = getCluster(positions)
        av_cluster_density = average_cluster_density(clusters)
        timestep.append(i)
        n_clusters.append(len(clusters))
        n_not_clustered.append(len(not_clustered))
        densities.append(av_cluster_density)
        mean, std = distance_mean_std(positions)
        means.append(mean)
        stds.append(std)
    cluster_data = [timestep, n_clusters, n_not_clustered]
    cluster_density_data = [timestep, densities]
    distance_data = [timestep, means, stds]
    cluster_plot(cluster_data, cluster_outfile)
    clusterdensity_plot(cluster_density_data, density_outfile)
    distance_plot(distance_data, distance_outfile)


def evaluate_velocities(infile, polarization_outfile):
    velocity_history = np.load(infile)
    timestep = []
    mean_pols = []
    std_pols = []
    for i, velocities in enumerate(velocity_history):
        timestep.append(i)
        mean_pol, std_pol = get_polarization(velocities)
        mean_pols.append(mean_pol)
        std_pols.append(std_pol)
    polarization_data = [timestep, mean_pols, std_pols]
    polarization_plot(polarization_data, polarization_outfile)
 



if __name__=='__main__':
    evaluate_positions("evaluation/rnn_positions.npy", "figures/rnn_density", "figures/rnn_clusters", "figures/rnn_distances")
    evaluate_positions("evaluation/couzin_positions.npy", "figures/couzin_density", "figures/couzin_clusters", "figures/couzin_distances")
    evaluate_velocities("evaluation/rnn_velocities.npy", "figures/rnn_polarization")
    evaluate_velocities("evaluation/couzin_velocities.npy", "figures/couzin_polarization")

