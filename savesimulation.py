from simulator import *
from helperfunctions import merge_data_sets
import os
import time
import datetime
import sys

num_simulations = 400
T = 70
N = 20
auto_merge = False
tracks = []
start_time = time.time()
for i in range(1, num_simulations + 1):

    agents = save_simulation(N, T)
    for agent in agents:
        tracks.append(agent.history)
    time_tmp = time.time()
    time_diff = time_tmp - start_time
    average_sim_time = time_diff / (i)
    sys.stdout.write('\r i: {}  progress: {}  estimated time: {}s'.format(i, round(100*i/num_simulations, 2), datetime.timedelta(seconds=average_sim_time*(num_simulations-i))))


np_tracks = np.asarray(tracks)
if auto_merge and os.path.exists("Tracks/Tracks.npy"):
    np.save("Tracks/Tracks2", np_tracks)
    merge_data_sets(["Tracks/Tracks.npy", "Tracks/Tracks2.npy"])
else:
    np.save("Tracks/Tracks", np_tracks)