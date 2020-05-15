import matplotlib.pyplot as plt
import numpy as np

from open_spiel.python.project.part_2 import path_file

########
# PLOT #
########
plt.plot(
    np.loadtxt("data/cfr_1000k_iterations_ITER_kuhn_poker.txt"),
    [i*10**3 for i in np.loadtxt("data/cfr_1000k_iterations_EXPL_kuhn_poker.txt")],
    label="CFR"
)
plt.xlabel("Iterations", fontweight="bold")
plt.ylabel("Exploitability (mbb/g)", fontweight="bold")
plt.loglog()
plt.title("Exploitability of CFR-agent for Kuhn poker", fontweight="bold")
plt.savefig(path_file.path
            + "project_plots_code/regular_cfr_plot/"
            + "cfr_kuhn"
            + path_file.plot_type)
plt.show()

