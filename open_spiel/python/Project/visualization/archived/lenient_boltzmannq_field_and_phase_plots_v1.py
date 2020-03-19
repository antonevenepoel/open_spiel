from functools import partial

import numpy as np

import matplotlib.pyplot as plt

from open_spiel.python.egt import dynamics
from open_spiel.python.egt import visualization
from open_spiel.python.project.dynamics_lenient_boltzmannq import dynamics_lb

# True for field plot, False for phase plot
PLOT_FLAG = False

#payoff_matrix_prisoners_dilemma = np.array([[[3,0],[5,1]],[[3,5],[0,1]]])
payoff_matrix_prisoners_dilemma = np.array([[[4,1],[3,3]],[[4,3],[1,3]]])
payoff_matrix_matching_pennies = np.array([[[1,-1],[-1,1]],[[-1,1],[1,-1]]])
payoff_stag_hunt = np.array([[[1, 0], [2 / 3, 2 / 3]], [[1, 2 / 3], [0, 2 / 3]]])
payoff_matrix_rock_paper_scissors = np.array([[[0.0, -0.25, 0.5], [0.25, 0.0, -0.05], [-0.5, 0.05, 0.0]],[[0.0, 0.25, -0.5], [-0.25, 0.0, 0.05], [0.5, -0.05, 0.0]]])


# Prisoner's Dilemma
fig = plt.figure(figsize=(10,10))
for i, K in zip(range(6),reversed([1,2,5,10,15,25])):
    dyn_stag_hunt = dynamics_lb.MultiPopulationDynamicsLB(payoff_stag_hunt, [partial(dynamics_lb.lenient_boltzmann, K=K, temperature=0.05)] * 2)
    ax1 = fig.add_subplot(int("23{}".format(i+1)), projection="2x2")
    ax1.quiver(dyn_stag_hunt) if PLOT_FLAG else ax1.streamplot(dyn_stag_hunt, linewidth="velocity")
    ax1.set_title("Stag Hunt, \K={}".format(round(K,2)), fontweight="bold")
    ax1.set(xlabel="Player 1: Pr(Hunt)",ylabel="Player 2: Pr(Hunt)")

fig = plt.figure(figsize=(10, 10))
for i, T in zip(range(6), reversed([0.000001, 0.05, 0.1, 0.25, 1, 10])):
    dyn_rock_paper_scissors = dynamics_lb.SinglePopulationDynamicsLB(payoff_matrix_rock_paper_scissors,
                                                                partial(dynamics_lb.lenient_boltzmann, K= 1, temperature=T))
    ax1 = fig.add_subplot(int("23{}".format(i + 1)), projection="3x3")
    ax1.quiver(
        dyn_rock_paper_scissors)   if PLOT_FLAG else ax1.streamplot(dyn_rock_paper_scissors,linewidth="velocity", color="velocity")
    ax1.set_title("Biased RPS, \u03C4={}".format(round(T, 2)), fontweight="bold")

fig = plt.figure(figsize=(10,10))
for i, T in zip(range(6),reversed([0.000001,0.05,0.1,0.25,1,10])):
    dyn_rock_paper_scissors = dynamics.SinglePopulationDynamics(payoff_matrix_rock_paper_scissors, partial(dynamics.boltzmannq,temperature=T))
    ax1 = fig.add_subplot(int("23{}".format(i+1)), projection="3x3")
    ax1.quiver(dyn_rock_paper_scissors) # if PLOT_FLAG else ax1.streamplot(dyn_rock_paper_scissors,linewidth="velocity", color="velocity")
    ax1.set_title("Biased RPS, \u03C4={}".format(round(T,2)), fontweight="bold")

plt.show()

