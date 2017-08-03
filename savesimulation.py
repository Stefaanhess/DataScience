from simulator import *

num_simulations = 10
T = 200
N = 30

tracks = []
for i in range(num_simulations):
    print("Simulation: ", i)
    agents = do_simulation(N, T, False)
    print(len(agents))
    for agent in agents:
        tracks.append(agent.history)

np_tracks = np.asarray(tracks)
print(np_tracks.shape)
np.save("Tracks", np_tracks)