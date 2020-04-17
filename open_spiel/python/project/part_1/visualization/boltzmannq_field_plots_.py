from functools import partial

import numpy as np

import matplotlib.pyplot as plt

from open_spiel.python.egt import dynamics
from open_spiel.python.egt import visualization
from open_spiel.python.project.part_1.visualization import paths


# True for field plot, False for phase plot
PLOT_FLAG = False
color = "black"
linewidth = None

payoff_matrix_prisoners_dilemma = np.array([[[3,0],[5,1]],[[3,5],[0,1]]])
payoff_matrix_matching_pennies = np.array([[[1,-1],[-1,1]],[[-1,1],[1,-1]]])
payoff_matrix_battle_of_the_sexes = np.array([[[2,0],[0,1]],[[1,0],[0,2]]])
payoff_matrix_rock_paper_scissors = np.array([[[0.0, -0.25, 0.5], [0.25, 0.0, -0.05], [-0.5, 0.05, 0.0]],[[0.0, 0.25, -0.5], [-0.25, 0.0, 0.05], [0.5, -0.05, 0.0]]])

# Prisoner's Dilemma
fig = plt.figure(figsize=(20,20))
for i, T in zip(range(4),[0.01,1,3,float("inf")]):
    dyn_prisoners_dilemma = dynamics.MultiPopulationDynamics(payoff_matrix_prisoners_dilemma, [partial(dynamics.boltzmannq,temperature=T)] * 2)
    ax1 = fig.add_subplot(4,4,i+1, projection="2x2")
    ax1.quiver(dyn_prisoners_dilemma) if PLOT_FLAG else ax1.streamplot(dyn_prisoners_dilemma,linewidth=linewidth, color=color)
    ax1.set_title("Prisoner's Dilemma, \u03C4={}".format(round(T,2)), fontweight="bold")
    ax1.set_xlabel("Player 1: Pr(Cooperate)", fontweight="bold")
    ax1.set_ylabel("Player 2: Pr(Cooperate)", fontweight="bold")

# Matching Pennies
# fig = plt.figure(figsize=(10,10))
for i, T in zip(range(4),[0.01,0.5,1,float("inf")]):
    dyn_matching_pennies = dynamics.MultiPopulationDynamics(payoff_matrix_matching_pennies, [partial(dynamics.boltzmannq,temperature=T)] * 2)
    ax1 = fig.add_subplot(4,4,i+1+4, projection="2x2")
    ax1.quiver(dyn_matching_pennies) if PLOT_FLAG else ax1.streamplot(dyn_matching_pennies,linewidth=linewidth, color=color)
    ax1.set_title("Matching Pennies, \u03C4={}".format(round(T,2)), fontweight="bold")
    ax1.set_xlabel("Player 1: Pr(Head)", fontweight="bold")
    ax1.set_ylabel("Player 2: Pr(Head)", fontweight="bold")

# Battle of the sexes
# fig = plt.figure(figsize=(10,10))
for i, T in zip(range(4),[0.01,0.5,0.75,float("inf")]):
    dyn_battle_of_the_sexes = dynamics.MultiPopulationDynamics(payoff_matrix_battle_of_the_sexes, [partial(dynamics.boltzmannq,temperature=T)] * 2)
    ax1 = fig.add_subplot(4,4,i+1+8, projection="2x2")
    ax1.quiver(dyn_battle_of_the_sexes) if PLOT_FLAG else ax1.streamplot(dyn_battle_of_the_sexes,linewidth=linewidth, color=color)
    ax1.set_title("Battle of the Sexes, \u03C4={}".format(round(T,2)), fontweight="bold")
    ax1.set_xlabel("Player 1: Pr(Boxing)", fontweight="bold")
    ax1.set_ylabel("Player 2: Pr(Boxing)", fontweight="bold")

# Biased rock, paper, scissors
# fig3 = plt.figure(figsize=(10,10))
for i, T in zip(range(4),[0.01,0.1,0.25,float("inf")]):
    dyn_rock_paper_scissors = dynamics.SinglePopulationDynamics(payoff_matrix_rock_paper_scissors, partial(dynamics.boltzmannq,temperature=T))
    ax1 = fig.add_subplot(4,4,i+1+12, projection="3x3")
    ax1.quiver(dyn_rock_paper_scissors) # if PLOT_FLAG else ax1.streamplot(dyn_rock_paper_scissors,linewidth="velocity", color="velocity")
    ax1.set_title("Biased RPS, \u03C4={}".format(round(T,2)), fontweight="bold")
    ax1.set_labels(["Rock", "Paper", "Scissors"])

plt.show()

# save
# `paths` voor persoonlijke paden
path = paths.path_arnout if paths.path_flag else paths.path_anton
fig.savefig(path + 'boltzmann' + '.' + paths.type)
