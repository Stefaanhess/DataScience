from simulator import *
from helperfunctions import merge_data_sets
import os
import time
import datetime
import sys
from Settings import Settings

# Settings
settings = Settings()
num_simulations = 100
T = 70
N = settings.N
file = settings.outfile
auto_merge = False
overwrite = False
tracks = []
start_time = time.time()

# Warning
if not auto_merge and overwrite and os.path.exists(file):
    print('')
    print("Caution: Exsisting Data will be Overwritten!!!")
    print('')

# Get Tracks
for i in range(1, num_simulations + 1):
    agents = save_simulation(N, T)
    for agent in agents:
        tracks.append(agent.history)
    time_tmp = time.time()
    time_diff = time_tmp - start_time
    average_sim_time = time_diff / (i)
    sys.stdout.write('\r i: {}  progress: {}  estimated time: {}s'.format(i, round(100*i/num_simulations, 2), datetime.timedelta(seconds=average_sim_time*(num_simulations-i))))

# Save:
np_tracks = np.asarray(tracks)
if auto_merge and os.path.exists(file):
    merge_tracks_to_file(file, np_tracks)
elif not overwrite and os.path.exists(file):
    np.save(file[:-4] + '__' + str(round(time.time())) + '.npy', np_tracks)
else:
    np.save(file, np_tracks)