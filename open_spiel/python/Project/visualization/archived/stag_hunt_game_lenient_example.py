from functools import partial

import numpy as np

import matplotlib.pyplot as plt

from open_spiel.python.egt import dynamics
from open_spiel.python.egt import visualization
from open_spiel.python.project.dynamics_lenient_boltzmannq import dynamics_lb
# True for field plot, False for phase plot
PLOT_FLAG = False

payoff_stag_hunt = np.array([[[1, 0], [2 / 3, 2 / 3]], [[1, 2 / 3], [0, 2 / 3]]])

# Stag Hunt
fig = plt.figure(figsize=(10,10))
for i, K in zip(range(6),reversed([0.001,1,2,5,10,1000])):
    dyn_stag_hunt = dynamics_lb.MultiPopulationDynamicsLB(payoff_stag_hunt, [partial(dynamics_lb.lenient_boltzmann, K= K, temperature=0.05)] * 2)
    ax1 = fig.add_subplot(int("23{}".format(i+1)), projection="2x2")
    ax1.quiver(dyn_stag_hunt) if PLOT_FLAG else ax1.streamplot(dyn_stag_hunt, linewidth="velocity", color="velocity")
    ax1.set_title("Stag Hunt, K={}".format(round(K,2)), fontweight="bold")
    ax1.set(xlabel="Player 1: Pr(Hunt)",ylabel="Player 2: Pr(Hunt)")

plt.show()