from helperfunctions import *

import numpy as np

# Couzin parameters
Rr = 0
Ro = 0.1
Ra = 200
max_turn = 90

def couzin_next_step(aj, agents, norm, noise=2):
    """Asses the next step for a single agent according to the Couzin paper."""
    dist, pos_dif = calculate_distances(aj, agents) 
    if (neighbours_in_zoneR(dist)==True):
        new_direct = calculate_direction_R(dist, pos_dif, norm)
        
    elif (neighbours_in_zoneO(dist)==True and neighbours_in_zoneA(dist)==True):
        new_direct = 0.5*(calculate_direction_A(dist, pos_dif, norm) + calculate_direction_O(dist, agents, norm))

    elif (neighbours_in_zoneO(dist)==True):
        new_direct = calculate_direction_O(dist, agents, norm)

    elif (neighbours_in_zoneA(dist)==True):
        new_direct = calculate_direction_A(dist, pos_dif, norm)

    else:
        new_direct = np.zeros(2)

    aj.velo_temp = normalize(aj.velo + new_direct, norm)
    aj.velo_temp = add_noise(aj.velo_temp, noise=noise)
    aj.velo_temp = set_angle(aj.velo, aj.velo_temp, norm, max_turn)


# Check if agents are in repulsion, orientation or attraction zone
def neighbours_in_zoneR(distances):
    for i in range(len(distances)):
        if(distances[i] > 0.0 and distances[i] < Rr):
            return(True)
    return(False)

def neighbours_in_zoneO(distances):
    for i in range(len(distances)):
        if(distances[i] >= Rr and distances[i] < Ro):
            return(True)
    return(False)
        
def neighbours_in_zoneA(distances):
    for i in range(len(distances)):
        if(distances[i] >= Ro and distances[i] < Ra):
            return(True)
    return(False)


# Calculate new direction according to zones of agents
def calculate_direction_R(dist, pos_dif, norm):
    direct = np.zeros(2)
    for i in range(len(dist)):
        if (dist[i] < Rr and dist[i] > 0.0):
            direct = direct + normalize((pos_dif[i]/dist[i]), norm)
    return direct

def calculate_direction_O(dist, agents, norm):
    direct = np.zeros(2)
    for i in range(len(dist)):
        if (dist[i] >= Rr and dist[i] < Ro):
            direct += normalize(agents[i].velo/np.linalg.norm(agents[i].velo), norm)
    return direct

def calculate_direction_A(dist, pos_dif, norm):
    direct = np.zeros(2)
    for i in range(len(dist)):
        if (dist[i] >= Ro and dist[i] < Ra):
            direct = direct - normalize((pos_dif[i]/dist[i]), norm)
    return direct
