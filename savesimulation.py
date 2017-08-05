from simulator import *

num_simulations = 20
T = 60
N = 20

tracks = []

for i in range(num_simulations):
    print("Simulation: ", i)
    agents = save_simulation(N, T)
    for agent in agents:
        tracks.append(agent.history[1:])


np_tracks = np.asarray(tracks)
print(np_tracks.shape)
np.save("Tracks", np_tracks)