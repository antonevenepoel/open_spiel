import numpy as np

import matplotlib.pyplot as plt

from open_spiel.python.egt import dynamics
from open_spiel.python.egt import visualization

payoff_matrix_prisoners_dilemma = np.array([[[3, 0],[5,1]],[[3,5],[0,1]]])
payoff_matrix_matching_pennies = np.array([[[1,0],[0,1]],[[0,1],[1,0]]])
payoff_matrix_battle_of_the_sexes = np.array([[[2,0],[0,1]],[[1,0],[0,2]]])
payoff_matrix_rock_paper_scissors = np.array([[[0,-1,1],[1,0,-1],[-1,1,0]],[[0,1,-1],[-1,0,1],[1,-1,0]]])

dyn_prisoners_dilemma = dynamics.MultiPopulationDynamics(payoff_matrix_prisoners_dilemma, [dynamics.replicator] * 2)
dyn_matching_pennies = dynamics.MultiPopulationDynamics(payoff_matrix_matching_pennies, [dynamics.replicator] * 2)
dyn_battle_of_the_sexes = dynamics.MultiPopulationDynamics(payoff_matrix_battle_of_the_sexes, [dynamics.replicator] * 2)
dyn_rock_paper_scissors = dynamics.SinglePopulationDynamics(payoff_matrix_rock_paper_scissors, dynamics.replicator)

#############
# Version 1 #
#############

# fig, ax = plt.subplots()
# X,Y,U,V = visualization._eval_dynamics_2x2_grid(dyn_prisoners_dilemma,num_points=9)
# ax.set_title("Prisoner's Dilemma")
# ax.set(xlabel="Player 1: Pr(Cooperate)",ylabel="Player 2: Pr(Cooperate)")
# q = ax.quiver(X, Y, U, V)
# plt.show()
#
# fig, ax = plt.subplots()
# X,Y,U,V = visualization._eval_dynamics_2x2_grid(dyn_matching_pennies,num_points=9)
# ax.set_title("Matching Pennies")
# ax.set(xlabel="Player 1: Pr(Playing Head)",ylabel="Player 2: Pr(Playing Head)")
# q = ax.quiver(X, Y, U, V)
# plt.show()
#
# fig, ax = plt.subplots()
# X,Y,U,V = visualization._eval_dynamics_2x2_grid(dyn_battle_of_the_sexes,num_points=9)
# ax.set_title("Battle of the Sexes (Boxing v. Shopping)")
# ax.set(xlabel="Player 1: Pr(Boxing)",ylabel="Player 2: Pr(Boxing)")
# q = ax.quiver(X, Y, U, V)
# plt.show()

#############
# Version 2 #
#############

# Quiver = directional field plot
# Streamplot = phase plot

#####
# 2x2: Prisoner's Dilemma, Matching Pennies, Battle of the Sexes (Population of 2)
#####

fig = plt.figure(figsize=(10,10))

ax1 = fig.add_subplot(231, projection="2x2")
ax1.quiver(dyn_prisoners_dilemma)
ax1.set_title("Prisoner's Dilemma", fontweight="bold")
ax1.set(xlabel="Player 1: Pr(Cooperate)",ylabel="Player 2: Pr(Cooperate)")
ax1a = fig.add_subplot(234, projection="2x2")
ax1a.streamplot(dyn_prisoners_dilemma, color="velocity", linewidth="velocity")
ax1a.set_title("Prisoner's Dilemma", fontweight="bold")
ax1a.set(xlabel="Player 1: Pr(Cooperate)",ylabel="Player 2: Pr(Cooperate)")

ax2 = fig.add_subplot(232, projection="2x2")
ax2.quiver(dyn_matching_pennies)
ax2.set_title("Matching Pennies", fontweight="bold")
ax2.set(xlabel="Player 1: Pr(Playing Head)",ylabel="Player 2: Pr(Playing Head)")
ax2a = fig.add_subplot(235, projection="2x2")
ax2a.streamplot(dyn_matching_pennies, color="velocity", linewidth="velocity")
ax2a.set_title("Matching Pennies", fontweight="bold")
ax2a.set(xlabel="Player 1: Pr(Playing Head)",ylabel="Player 2: Pr(Playing Head)")

ax3 = fig.add_subplot(233, projection="2x2")
ax3.quiver(dyn_battle_of_the_sexes)
ax3.set_title("Battle of the Sexes", fontweight="bold")
ax3.set(xlabel="Player 1: Pr(Boxing)",ylabel="Player 2: Pr(Boxing)")
ax3a = fig.add_subplot(236, projection="2x2")
ax3a.streamplot(dyn_battle_of_the_sexes, color="velocity", linewidth="velocity")
ax3a.set_title("Battle of the Sexes", fontweight="bold")
ax3a.set(xlabel="Player 1: Pr(Boxing)",ylabel="Player 2: Pr(Boxing)")

#####
# 3x3: Rock, Paper, Scissors (Single population)
#####

# TODO: Labels of Rock, Paper and Scissor

fig1 = plt.figure(figsize=(10,10))

ax = fig1.add_subplot(121, projection="3x3")
ax.quiver(dyn_rock_paper_scissors, boundary=True)
ax.set_title("Rock, Paper, Scissors", fontweight="bold")
ax1 = fig1.add_subplot(122, projection="3x3")
ax1.streamplot(dyn_rock_paper_scissors, color="velocity", linewidth="velocity")
ax1.set_title("Rock, Paper, Scissors", fontweight="bold")

plt.show()