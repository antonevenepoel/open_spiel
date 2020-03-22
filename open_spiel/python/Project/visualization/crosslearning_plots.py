import numpy as np

import matplotlib.pyplot as plt

from open_spiel.python.egt import dynamics
from open_spiel.python.egt import visualization
from open_spiel.python.Project.visualization.probarray_visualization import prepare_plot

payoff_matrix_prisoners_dilemma = np.array([[[3,0],[5,1]],[[3,5],[0,1]]])
payoff_matrix_matching_pennies = np.array([[[1,-1],[-1,1]],[[-1,1],[1,-1]]])
payoff_matrix_battle_of_the_sexes = np.array([[[2,0],[0,1]],[[1,0],[0,2]]])
payoff_matrix_rock_paper_scissors = np.array([[[0.0, -0.25, 0.5], [0.25, 0.0, -0.05], [-0.5, 0.05, 0.0]],[[0.0, 0.25, -0.5], [-0.25, 0.0, 0.05], [0.5, -0.05, 0.0]]])

dyn_prisoners_dilemma = dynamics.MultiPopulationDynamics(payoff_matrix_prisoners_dilemma, [dynamics.replicator] * 2)
dyn_matching_pennies = dynamics.MultiPopulationDynamics(payoff_matrix_matching_pennies, [dynamics.replicator] * 2)
dyn_battle_of_the_sexes = dynamics.MultiPopulationDynamics(payoff_matrix_battle_of_the_sexes, [dynamics.replicator] * 2)
dyn_rock_paper_scissors = dynamics.SinglePopulationDynamics(payoff_matrix_rock_paper_scissors, dynamics.replicator)

# IMPORTANT REMARK: The accuracy of the probarray_visualization needs to be set at 1.

pd = False
bos = False
mp = True

if (pd):
    probs_prisonners_dilemma1 = prepare_plot("PD", (0.5,0.5), 2)
    probs_prisonners_dilemma2 = prepare_plot("PD", (0.8,0.2), 2)
    probs_prisonners_dilemma3 = prepare_plot("PD", (0.2,0.3), 2)
    probs_prisonners_dilemma4 = prepare_plot("PD", (0.6,0.2), 2)
    probs_prisonners_dilemma5 = prepare_plot("PD", (0.8,0.6), 2)

if(bos):
    probs_battle_of_the_sexes1 = prepare_plot("BOS", (0.5, 0.5), 2)
    probs_battle_of_the_sexes2 = prepare_plot("BOS", (0.8, 0.8), 2)
    probs_battle_of_the_sexes3 = prepare_plot("BOS", (0.6, 0.3), 2)
    probs_battle_of_the_sexes4 = prepare_plot("BOS", (0.5, 0.7), 2)
    probs_battle_of_the_sexes5 = prepare_plot("BOS", (0.6, 0.5), 2)


if(mp):
    probs_matching_pennies1 = prepare_plot("MP", (0.5, 0.5), 2)
    probs_matching_pennies2 = prepare_plot("MP", (0.6, 0.6), 2)
    probs_matching_pennies3 = prepare_plot("MP", (0.7, 0.7), 2)
    probs_matching_pennies4 = prepare_plot("MP", (0.8, 0.8), 2)
    probs_matching_pennies5 = prepare_plot("MP", (0.75, 0.75), 2)


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

fig1 = plt.figure(figsize=(10,10))

ax1 = fig1.add_subplot(231, projection="2x2")
ax1.quiver(dyn_prisoners_dilemma)
ax1.set_title("Prisoner's Dilemma: average trajectory of Q-Learning", fontweight="bold")
ax1.set(xlabel="Player 1: Pr(Cooperate)",ylabel="Player 2: Pr(Cooperate)")
if(pd):
    ax1.plot(probs_prisonners_dilemma1)
    ax1.plot(probs_prisonners_dilemma2)
    ax1.plot(probs_prisonners_dilemma3)
    ax1.plot(probs_prisonners_dilemma4)
    ax1.plot(probs_prisonners_dilemma5)

fig2 = plt.figure(figsize=(10, 10))
ax1a = fig2.add_subplot(234, projection="2x2")
ax1a.streamplot(dyn_prisoners_dilemma, color="velocity", linewidth="velocity")
ax1a.set_title("Prisoner's Dilemma", fontweight="bold")
ax1a.set(xlabel="Player 1: Pr(Cooperate)",ylabel="Player 2: Pr(Cooperate)")

ax2 = fig2.add_subplot(232, projection="2x2")
ax2.quiver(dyn_matching_pennies)
ax2.set_title("Matching Pennies", fontweight="bold")
ax2.set(xlabel="Player 1: Pr(Playing Head)",ylabel="Player 2: Pr(Playing Head)")
if(mp):
    ax2.plot(probs_matching_pennies1)
    ax2.plot(probs_matching_pennies2)
    ax2.plot(probs_matching_pennies3)
    ax2.plot(probs_matching_pennies4)
    ax2.plot(probs_matching_pennies5)
ax2a = fig2.add_subplot(235, projection="2x2")
ax2a.streamplot(dyn_matching_pennies, color="velocity", linewidth="velocity")
ax2a.set_title("Matching Pennies", fontweight="bold")
ax2a.set(xlabel="Player 1: Pr(Playing Head)",ylabel="Player 2: Pr(Playing Head)")

ax3 = fig2.add_subplot(233, projection="2x2")
ax3.quiver(dyn_battle_of_the_sexes)
ax3.set_title("Battle of the Sexes", fontweight="bold")
ax3.set(xlabel="Player 1: Pr(Boxing)",ylabel="Player 2: Pr(Boxing)")
if(bos):
    ax3.plot(probs_battle_of_the_sexes1)
    ax3.plot(probs_battle_of_the_sexes2)
    ax3.plot(probs_battle_of_the_sexes3)
    ax3.plot(probs_battle_of_the_sexes4)
    ax3.plot(probs_battle_of_the_sexes5)
#ax3.plot(probs_battle_of_the_sexes, color=color, marker=marker, linestyle=linestyle)
ax3a = fig2.add_subplot(236, projection="2x2")
ax3a.streamplot(dyn_battle_of_the_sexes, color="velocity", linewidth="velocity")
ax3a.set_title("Battle of the Sexes", fontweight="bold")
ax3a.set(xlabel="Player 1: Pr(Boxing)",ylabel="Player 2: Pr(Boxing)")

plt.show()



