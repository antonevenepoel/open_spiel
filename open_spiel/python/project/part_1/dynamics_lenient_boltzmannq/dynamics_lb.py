"""
This module is an extension of the dynamics module available in open_spiel/python/egt.
"""
import numpy as np
from open_spiel.python.egt import dynamics



def utilities_vector(payOff, stateX, stateY, K):
    size = stateX.shape[0]

    utilities = np.zeros(size)

    for i in range(size):
        for j in range(size):
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

class SinglePopulationDynamicsLB(object):
  """Continuous-time single population dynamics.

  Attributes:
    payoff_matrix: The payoff matrix as an `numpy.ndarray` of shape `[2, k_1,
      k_2]`, where `k_1` is the number of strategies of the first player and
      `k_2` for the second player. The game is assumed to be symmetric.
    dynamics: A callback function that returns the time-derivative of the
      population state.
  """

  def __init__(self, payoff_matrix, dynamics):
    """Initializes the single-population dynamics."""
    assert payoff_matrix.ndim == 3
    assert payoff_matrix.shape[0] == 2
    assert np.allclose(payoff_matrix[0], payoff_matrix[1].T)
    self.payoff_matrix = payoff_matrix[0]
    self.dynamics = dynamics

  def __call__(self, state=None, time=None):
    """Time derivative of the population state.

    Args:
      state: Probability distribution as list or
        `numpy.ndarray(shape=num_strategies)`.
      time: Time is ignored (time-invariant dynamics). Including the argument in
        the function signature supports numerical integration via e.g.
        `scipy.integrate.odeint` which requires that the callback function has
        at least two arguments (state and time).

    Returns:
      Time derivative of the population state as
      `numpy.ndarray(shape=num_strategies)`.
    """
    state = np.array(state)
    assert state.ndim == 1
    assert state.shape[0] == self.payoff_matrix.shape[0]
    # (Ax')' = xA'
    fitness = np.matmul(state, self.payoff_matrix.T)
    return self.dynamics(self.payoff_matrix , state, state)


class MultiPopulationDynamicsLB(object):
  """Continuous-time multi-population dynamics.

  Attributes:
    payoff_tensor: The payoff tensor as an numpy.ndarray of size `[n, k0, k1,
      k2, ...]`, where n is the number of players and `k0` is the number of
      strategies of the first player, `k1` of the second player and so forth.
    dynamics: List of callback functions for the time-derivative of the
      population states, where `dynamics[i]` computes the time-derivative of the
      i-th player's population state. If at construction, only a single callback
      function is provided, the same function is used for all populations.
  """

  def __init__(self, payoff_tensor, dynamics):
    """Initializes the multi-population dynamics."""
    if isinstance(dynamics, list) or isinstance(dynamics, tuple):
      assert payoff_tensor.shape[0] == len(dynamics)
    else:
      dynamics = [dynamics] * payoff_tensor.shape[0]
    self.payoff_tensor = payoff_tensor
    self.dynamics = dynamics

  def __call__(self, state, time=None):
    """Time derivative of the population states.

    Args:
      state: Combined population state for all populations as a list or flat
        `numpy.ndarray` (ndim=1). Probability distributions are concatenated in
        order of the players.
      time: Time is ignored (time-invariant dynamics). Including the argument in
        the function signature supports numerical integration via e.g.
        `scipy.integrate.odeint` which requires that the callback function has
        at least two arguments (state and time).

    Returns:
      Time derivative of the combined population state as `numpy.ndarray`.
    """
    state = np.array(state)
    # cumsum: Return the cumulative sum of the elements along a given axis.
    n = self.payoff_tensor.shape[0]  # number of players
    ks = self.payoff_tensor.shape[1:]  # number of strategies for each player
    assert state.shape[0] == sum(ks)

    states = np.split(state, np.cumsum(ks)[:-1]) #Split the state in an array of states for each player.
    dstates = [None] * n
    for i in range(n):
      # move i-th population to front
      payOff = np.moveaxis(self.payoff_tensor[i], i, 0) #Choose the payoff matrix of player i
      # marginalize out all other populations
      for i_ in set(range(n)) - {i}:
        other_state = states[i_]
      dstates[i] = self.dynamics[i](payOff, states[i], other_state)

    return np.concatenate(dstates)




# TEST
# payOff = np.array([[3,0],[5,1]])
# X = np.array([0.5, 0.5])
# Y = np.array([0.8, 0.2])
#
# print(lenient_boltzmann(payOff, X, Y, 3))

#
# payOff = np.array([[11,-30, 0],[-30,7,6],[0,0,5]])
# X = np.array([0.3333333, 0.33333, 0.3333])
# Y = np.array([0.3333, 0.3333, 0.33333])
# print(utilities_vector(payOff, X, Y, 2))