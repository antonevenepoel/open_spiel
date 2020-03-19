
import numpy as np
from open_spiel.python.egt import dynamics

def utilities_vector(payOff, stateX, stateY, K):
    size = stateX.shape[0]
    utilities = np.zeros(size)
    for i in range(size):
        for j in range(size-1):
            utilities[i] += payOff[i,j]*stateY[j]*(collector1(i,j,size, stateY,payOff)**K - collector2(i,j,size, stateY,payOff)**K)/collector3(i,j,size, stateY,payOff)
    return utilities


def collector1(i, j, size, stateY, payOff):
    reward = 0
    for k in range(size):
        if payOff[i,k] <= payOff[i,j]:
            reward +=stateY[k]
    return reward

def collector2(i, j, size, stateY, payOff):
    reward = 0
    for k in range(size):
        if payOff[i,k] < payOff[i,j]:
            reward +=stateY[k]
    return reward

def collector3(i, j, size, stateY, payOff):
    reward = 0
    for k in range(size):
        if payOff[i,k] == payOff[i,j]:
            reward +=stateY[k]
    return reward

def lenient_boltzmann(payOff, stateX, stateY, K, temperature):
    fitness = utilities_vector(payOff, stateX, stateY, K)
    return dynamics.boltzmannq(stateX, fitness, temperature)



# TEST
# payOff = np.array([[3,0],[5,1]])
# X = np.array([0.5, 0.5])
# Y = np.array([0.8, 0.2])
#
# print(lenient_boltzmann(payOff, X, Y, 3))