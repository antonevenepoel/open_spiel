import numpy as np

import matplotlib.pyplot as plt

from open_spiel.python.egt import dynamics
from open_spiel.python.egt import visualization
from open_spiel.python.project.visualization import paths
from open_spiel.python.project.visualization.probarray_visualization import prepare_plot

payoff_matrix_prisoners_dilemma = np.array([[[3,0],[5,1]],[[3,5],[0,1]]])
payoff_matrix_matching_pennies = np.array([[[1,-1],[-1,1]],[[-1,1],[1,-1]]])
payoff_matrix_battle_of_the_sexes = np.array([[[2,0],[0,1]],[[1,0],[0,2]]])
payoff_matrix_rock_paper_scissors = np.array([[[0.0, -0.25, 0.5], [0.25, 0.0, -0.05], [-0.5, 0.05, 0.0]],[[0.0, 0.25, -0.5], [-0.25, 0.0, 0.05], [0.5, -0.05, 0.0]]])

dyn_prisoners_dilemma = dynamics.MultiPopulationDynamics(payoff_matrix_prisoners_dilemma, [dynamics.replicator] * 2)
dyn_matching_pennies = dynamics.MultiPopulationDynamics(payoff_matrix_matching_pennies, [dynamics.replicator] * 2)
dyn_battle_of_the_sexes = dynamics.MultiPopulationDynamics(payoff_matrix_battle_of_the_sexes, [dynamics.replicator] * 2)
dyn_rock_paper_scissors = dynamics.SinglePopulationDynamics(payoff_matrix_rock_paper_scissors, dynamics.replicator)

# IMPORTANT REMARK: The accuracy of the probarray_visualization needs to be set at 1.


PLOT_FLAG = True

pd = True
bos = True
mp = True
rps = True
if (pd):
    probs_prisonners_dilemma1 = prepare_plot("PD", (0.7,0.7), 2)
    probs_prisonners_dilemma2 = prepare_plot("PD", (0.9,0.5), 2)
    probs_prisonners_dilemma3 = prepare_plot("PD", (0.5,0.9), 2)
    probs_prisonners_dilemma4 = prepare_plot("PD", (0.7,0.4), 2)
    probs_prisonners_dilemma5 = prepare_plot("PD", (0.4,0.7), 2)

if(mp):
    probs_matching_pennies1 = prepare_plot("MP", (0.55, 0.55), 2)
    probs_matching_pennies2 = prepare_plot("MP", (0.75, 0.75), 2)
    probs_matching_pennies3 = prepare_plot("MP", (0.9, 0.9), 2)

if(rps):
    probs_rock_paper_scissors1 = prepare_plot("RPS", ((0.3 , 0.3), (0.3 , 0.3)), 3)
    probs_rock_paper_scissors2 = prepare_plot("RPS", ((0.8, 0.1), (0.8, 0.1)), 3)
    # probs_rock_paper_scissors3 = prepare_plot("RPS", ((0.1, 0.8), (0.1, 0.8)), 3)
    probs_rock_paper_scissors4 = prepare_plot("RPS", ((0.1, 0.1), (0.1, 0.1)), 3)

if(bos):
    probs_battle_of_the_sexes1 = prepare_plot("BOS", (0.3, 0.6), 2)
    probs_battle_of_the_sexes2 = prepare_plot("BOS", (1-0.6, 1-0.3), 2)
    probs_battle_of_the_sexes3 = prepare_plot("BOS", (0.2, 0.4), 2)
    probs_battle_of_the_sexes4 = prepare_plot("BOS", (1-0.4, 1-0.2), 2)
    probs_battle_of_the_sexes5 = prepare_plot("BOS", (0.8, 0.1), 2)
    probs_battle_of_the_sexes6 = prepare_plot("BOS", (1-0.1, 1-0.8), 2)


# Opmaak voor de trace plots
color = "black"
marker = "D"
linestyle = "dashed"

#############
# Version 2 #
#############

# Quiver = directional field plot
# Streamplot = phase plot

#####
# 2x2: Prisoner's Dilemma, Matching Pennies, Battle of the Sexes (Population of 2)
#####

fig1 = plt.figure(figsize=(12,4))


ax1 = fig1.add_subplot(141, projection="2x2")
ax1.quiver(dyn_prisoners_dilemma) if PLOT_FLAG else ax1.streamplot(dyn_prisoners_dilemma,linewidth="velocity", color="velocity")
ax1.set_title("Prisoner's Dilemma", fontweight="bold")
ax1.set_xlabel("Player 1: Pr(Cooperate)", fontweight="bold")
ax1.set_ylabel("Player 2: Pr(Cooperate)", fontweight="bold")

if(pd):
    ax1.plot(probs_prisonners_dilemma1, color=color)
    ax1.plot(probs_prisonners_dilemma2, color=color)
    ax1.plot(probs_prisonners_dilemma3, color=color)
    ax1.plot(probs_prisonners_dilemma4, color=color)
    ax1.plot(probs_prisonners_dilemma5, color=color)


ax2 = fig1.add_subplot(142, projection="2x2")
ax2.quiver(dyn_matching_pennies) if PLOT_FLAG else ax2.streamplot(dyn_matching_pennies,linewidth="velocity", color="velocity")
ax2.set_title("Matching Pennies", fontweight="bold")
ax2.set_xlabel("Player 1: Pr(Playing Head)", fontweight="bold")
ax2.set_ylabel("Player 2: Pr(Playing Head)", fontweight="bold")
if(mp):
    ax2.plot(probs_matching_pennies1, color=color)
    ax2.plot(probs_matching_pennies2, color=color)
    ax2.plot(probs_matching_pennies3, color=color)


ax3 = fig1.add_subplot(143, projection="2x2")
ax3.quiver(dyn_battle_of_the_sexes) if PLOT_FLAG else ax3.streamplot(dyn_battle_of_the_sexes,linewidth="velocity", color="velocity")
ax3.set_title("Battle of the Sexes", fontweight="bold")
ax3.set_xlabel("Player 1: Pr(Boxing)", fontweight="bold")
ax3.set_ylabel("Player 2: Pr(Boxing)", fontweight="bold")
if(bos):
    ax3.plot(probs_battle_of_the_sexes1, color=color)
    ax3.plot(probs_battle_of_the_sexes2, color=color)
    ax3.plot(probs_battle_of_the_sexes3, color=color)
    ax3.plot(probs_battle_of_the_sexes4, color=color)
    ax3.plot(probs_battle_of_the_sexes5, color=color)
    ax3.plot(probs_battle_of_the_sexes6, color=color)


ax = fig1.add_subplot(144, projection="3x3")
ax.quiver(dyn_rock_paper_scissors, boundary=True)
ax.set_title("Biased Rock, Paper, Scissors", fontweight="bold")
ax.set_labels(["Rock", "Paper", "Scissors"])
if rps:
    ax.plot(probs_rock_paper_scissors1, color=color)
    ax.plot (probs_rock_paper_scissors2, color=color)
    # ax.plot(probs_rock_paper_scissors3, color=color)
    ax.plot(probs_rock_paper_scissors4, color=color)


plt.show()

# save
# `paths` voor persoonlijke paden
path = paths.path_arnout if paths.path_flag else paths.path_anton
fig1.savefig(path + 'crosslearning' + '.' + paths.type)


