from functools import partial

import numpy as np

import matplotlib.pyplot as plt

from open_spiel.python.egt import dynamics
from open_spiel.python.egt import visualization
from open_spiel.python.project.visualization import paths


# True for field plot, False for phase plot
PLOT_FLAG = False
color = "black"

payoff_matrix_prisoners_dilemma = np.array([[[3,0],[5,1]],[[3,5],[0,1]]])
payoff_matrix_matching_pennies = np.array([[[1,-1],[-1,1]],[[-1,1],[1,-1]]])
payoff_matrix_battle_of_the_sexes = np.array([[[2,0],[0,1]],[[1,0],[0,2]]])
payoff_matrix_rock_paper_scissors = np.array([[[0.0, -0.25, 0.5], [0.25, 0.0, -0.05], [-0.5, 0.05, 0.0]],[[0.0, 0.25, -0.5], [-0.25, 0.0, 0.05], [0.5, -0.05, 0.0]]])

# Prisoner's Dilemma
fig = plt.figure(figsize=(10,10))
for i, T in zip(range(6),reversed([0.000001,1,2,3,4,float("inf")])):
    dyn_prisoners_dilemma = dynamics.MultiPopulationDynamics(payoff_matrix_prisoners_dilemma, [partial(dynamics.boltzmannq,temperature=T)] * 2)
    ax1 = fig.add_subplot(int("23{}".format(i+1)), projection="2x2")
    ax1.quiver(dyn_prisoners_dilemma) if PLOT_FLAG else ax1.streamplot(dyn_prisoners_dilemma,linewidth="velocity", color=color)
    ax1.set_title("Prisoner's Dilemma, \u03C4={}".format(round(T,2)), fontweight="bold")
    ax1.set(xlabel="Player 1: Pr(Cooperate)",ylabel="Player 2: Pr(Cooperate)")

# Matching Pennies
fig1 = plt.figure(figsize=(10,10))
for i, T in zip(range(6),reversed([0.000001,0.1,0.5,1,2,float("inf")])):
    dyn_matching_pennies = dynamics.MultiPopulationDynamics(payoff_matrix_matching_pennies, [partial(dynamics.boltzmannq,temperature=T)] * 2)
    ax1 = fig1.add_subplot(int("23{}".format(i+1)), projection="2x2")
    ax1.quiver(dyn_matching_pennies) if PLOT_FLAG else ax1.streamplot(dyn_matching_pennies,linewidth="velocity", color=color)
    ax1.set_title("Matching Pennies, \u03C4={}".format(round(T,2)), fontweight="bold")
    ax1.set(xlabel="Player 1: Pr(Playing Head)", ylabel="Player 2: Pr(Playing Head)")

# Battle of the sexes
fig2 = plt.figure(figsize=(10,10))
for i, T in zip(range(6),reversed([0.000001,0.5,0.75,1,2,float("inf")])):
    dyn_battle_of_the_sexes = dynamics.MultiPopulationDynamics(payoff_matrix_battle_of_the_sexes, [partial(dynamics.boltzmannq,temperature=T)] * 2)
    ax1 = fig2.add_subplot(int("23{}".format(i+1)), projection="2x2")
    ax1.quiver(dyn_battle_of_the_sexes) if PLOT_FLAG else ax1.streamplot(dyn_battle_of_the_sexes,linewidth="velocity", color=color)
    ax1.set_title("Battle of the Sexes, \u03C4={}".format(round(T,2)), fontweight="bold")
    ax1.set(xlabel="Player 1: Pr(Boxing)",ylabel="Player 2: Pr(Boxing)")

# Biased rock, paper, scissors
fig3 = plt.figure(figsize=(10,10))
for i, T in zip(range(6),reversed([0.000001,0.05,0.1,0.25,1,float("inf")])):
    dyn_rock_paper_scissors = dynamics.SinglePopulationDynamics(payoff_matrix_rock_paper_scissors, partial(dynamics.boltzmannq,temperature=T))
    ax1 = fig3.add_subplot(int("23{}".format(i+1)), projection="3x3")
    ax1.quiver(dyn_rock_paper_scissors) # if PLOT_FLAG else ax1.streamplot(dyn_rock_paper_scissors,linewidth="velocity", color="velocity")
    ax1.set_title("Biased RPS, \u03C4={}".format(round(T,2)), fontweight="bold")
    ax1.set_labels(["Rock", "Paper", "Scissors"])

plt.show()

# save
# `paths` voor persoonlijke paden
path = paths.path_arnout if paths.path_flag else paths.path_anton
fig.savefig(path + 'boltzmann-pd' + '.' + paths.type)
fig1.savefig(path + 'boltzmann-mp' + '.' + paths.type)
fig2.savefig(path + 'boltzmann-bos' + '.' + paths.type)
fig3.savefig(path + 'boltzmann-rps' + '.' + paths.type)
