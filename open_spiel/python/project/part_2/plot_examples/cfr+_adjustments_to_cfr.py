import matplotlib.pyplot as plt
import numpy as np

from open_spiel.python.project.part_2 import path_file
from open_spiel.python.project.part_2.plot import calculate_store_plot_cfr

###########
# EXAMPLE #
###########

plt.plot(
    np.loadtxt("data/cfr_10k_iterations_ITER7.txt"),
    np.loadtxt("data/cfr_10k_iterations_EXPL7.txt"),
    label="CFR"
)
plt.plot(
    np.loadtxt("data/cfr_10k_iterations_ITER10.txt"),
    np.loadtxt("data/cfr_10k_iterations_EXPL10.txt"),
    label="CFR and alternating updates"
)
plt.plot(
    np.loadtxt("data/cfr_10k_iterations_ITER8.txt"),
    np.loadtxt("data/cfr_10k_iterations_EXPL8.txt"),
    label="CFR and linear averaging"
)
plt.plot(
    np.loadtxt("data/cfr_10k_iterations_ITER9.txt"),
    np.loadtxt("data/cfr_10k_iterations_EXPL9.txt"),
    label="CFR and regret matching+"
)
plt.plot(
    np.loadtxt("data/cfr_10k_iterations_ITER11.txt"),
    np.loadtxt("data/cfr_10k_iterations_EXPL11.txt"),
    label="CFR+"
)
plt.xlabel("Iterations", fontweight="bold")
plt.ylabel("Exploitability", fontweight="bold")
plt.loglog()
plt.legend()
plt.title("Comparison of individual CFR+ adjustments to CFR in Leduc poker", fontweight="bold")
# plt.savefig(path_file.plot_path
#            + "cfr_adjustments"
#            + path_file.plot_type,
#            bbox_inches="tight")
plt.show()
