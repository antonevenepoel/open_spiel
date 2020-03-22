from functools import partial

import numpy as np

import matplotlib.pyplot as plt

from open_spiel.python.egt import dynamics
from open_spiel.python.egt import visualization
from open_spiel.python.project.dynamics_lenient_boltzmannq import dynamics_lb

# True for field plot, False for phase plot
from open_spiel.python.project.visualization import paths

PLOT_FLAG = False
figsize = (20,20)
color = "black"
linewidth = None

payoff_stag_hunt = np.array([[[4,1],[3,3]],[[4,3],[1,3]]])
payoff_matrix_prisoners_dilemma = np.array([[[3,0],[5,1]],[[3,5],[0,1]]])
payoff_matrix_matching_pennies = np.array([[[1,-1],[-1,1]],[[-1,1],[1,-1]]])
payoff_matrix_battle_of_the_sexes = np.array([[[2,0],[0,1]],[[1,0],[0,2]]])
payoff_matrix_rock_paper_scissors = np.array([[[0.0, -0.25, 0.5], [0.25, 0.0, -0.05], [-0.5, 0.05, 0.0]],[[0.0, 0.25, -0.5], [-0.25, 0.0, 0.05], [0.5, -0.05, 0.0]]])


# Stag Hunt
# fig = plt.figure(figsize=figsize)
# for i, K in zip(range(4),[1,3,10,float("inf")]):
#     dyn_stag_hunt = dynamics_lb.MultiPopulationDynamicsLB(payoff_stag_hunt, [partial(dynamics_lb.lenient_boltzmann,temperature=0.01,K=K)] * 2)
#     ax1 = fig.add_subplot(int("14{}".format(i+1)), projection="2x2")
#     ax1.quiver(dyn_stag_hunt) if PLOT_FLAG else ax1.streamplot(dyn_stag_hunt,linewidth="velocity", color="velocity")
#     ax1.set_title("Stag Hunt, \u03Ba={}".format(round(K,2)), fontweight="bold")
#     ax1.set(xlabel="Player 1: Pr(Stag)",ylabel="Player 2: Pr(Stag)")

# Prisoner's Dilemma
fig = plt.figure(figsize=figsize)
for i, K in zip(range(2),[1,float("inf")]):
    dyn_prisoners_dilemma = dynamics_lb.MultiPopulationDynamicsLB(payoff_matrix_prisoners_dilemma, [partial(dynamics_lb.lenient_boltzmann,temperature=0.01,K=K)] * 2)
    ax1 = fig.add_subplot(4,4,i+1,projection="2x2")
    ax1.quiver(dyn_prisoners_dilemma) if PLOT_FLAG else ax1.streamplot(dyn_prisoners_dilemma,linewidth=linewidth, color=color)
    ax1.set_title("Prisoner's Dilemma, \u03Ba={}".format(round(K,2)), fontweight="bold")
    ax1.set_xlabel("Player 1: Pr(Cooperate)", fontweight="bold")
    ax1.set_ylabel("Player 2: Pr(Cooperate)", fontweight="bold")

# Matching Pennies
# fig2 = plt.figure(figsize=figsize)
for i, K in zip(range(4),[1,7,20,float("inf")]):
    dyn_matching_pennies = dynamics_lb.MultiPopulationDynamicsLB(payoff_matrix_matching_pennies, [partial(dynamics_lb.lenient_boltzmann,temperature=0.01,K=K)] * 2)
    ax1 = fig.add_subplot(4,4,i+1+4, projection="2x2")
    ax1.quiver(dyn_matching_pennies) if PLOT_FLAG else ax1.streamplot(dyn_matching_pennies,linewidth=linewidth, color=color)
    ax1.set_title("Matching Pennies, \u03Ba={}".format(round(K,2)), fontweight="bold")
    ax1.set_xlabel("Player 1: Pr(Head)", fontweight="bold")
    ax1.set_ylabel("Player 2: Pr(Head)", fontweight="bold")

# Battle of the sexes
# fig3 = plt.figure(figsize=figsize)
for i, K in zip(range(4),[1,3,10,float("inf")]):
    dyn_battle_of_the_sexes = dynamics_lb.MultiPopulationDynamicsLB(payoff_matrix_battle_of_the_sexes, [partial(dynamics_lb.lenient_boltzmann,temperature=0.01,K=K)] * 2)
    ax1 = fig.add_subplot(4,4,i+1+8, projection="2x2")
    ax1.quiver(dyn_battle_of_the_sexes) if PLOT_FLAG else ax1.streamplot(dyn_battle_of_the_sexes,linewidth=linewidth, color=color)
    ax1.set_title("Battle of the Sexes, \u03Ba={}".format(round(K,2)), fontweight="bold")
    ax1.set_xlabel("Player 1: Pr(Boxing)", fontweight="bold")
    ax1.set_ylabel("Player 2: Pr(Boxing)", fontweight="bold")

# Biased rock, paper, scissors
# fig4 = plt.figure(figsize=figsize)
for i, K in zip(range(4),[1,5,10,float("inf")]):
    dyn_rock_paper_scissors = dynamics_lb.SinglePopulationDynamicsLB(payoff_matrix_rock_paper_scissors, partial(dynamics_lb.lenient_boltzmann,temperature=0.01,K=K))
    ax1 = fig.add_subplot(4,4,i+1+12,projection="3x3")
    ax1.quiver(dyn_rock_paper_scissors) # if PLOT_FLAG else ax1.streamplot(dyn_rock_paper_scissors,linewidth="velocity", color="velocity")
    ax1.set_title("Biased RPS, \u03Ba={}".format(round(K,2)), fontweight="bold")
    ax1.set_labels(["Rock", "Paper", "Scissors"])

plt.show()

# save
# `paths.py` voor persoonlijke paden
path = paths.path_arnout if paths.path_flag else paths.path_anton
# fig.savefig(path + 'lenient-sh' + '.' + paths.type)
fig.savefig(path + 'lenient-all' + '.' + paths.type)
