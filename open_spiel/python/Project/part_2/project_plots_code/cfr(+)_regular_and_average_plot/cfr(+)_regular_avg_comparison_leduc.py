import matplotlib.pyplot as plt
import numpy as np

from open_spiel.python.project.part_2 import path_file

########
# PLOT #
########
plt.plot(
    np.loadtxt("leduc_data/cfr_1k_iterations_ITER0_avg.txt"),
    [i*10**3 for i in np.loadtxt("leduc_data/cfr_1k_iterations_EXPL0_avg.txt")],
    label="CFR average strategy"
)
plt.plot(
    np.loadtxt("leduc_data/cfr_1k_iterations_ITER_leduc_poker.txt"),
    [i*10**3 for i in np.loadtxt("leduc_data/cfr_1k_iterations_EXPL_leduc_poker.txt")],
    label="CFR current strategy"
)
plt.plot(
    np.loadtxt("leduc_data/cfr_1k_iterations_ITER4_avg.txt"),
    [i*10**3 for i in np.loadtxt("leduc_data/cfr_1k_iterations_EXPL4_avg.txt")],
    label="CFR+ average strategy"
)
plt.plot(
    np.loadtxt("leduc_data/cfr_1k_iterations_ITER_leduc_poker0.txt"),
    [i*10**3 for i in np.loadtxt("leduc_data/cfr_1k_iterations_EXPL_leduc_poker0.txt")],
    label="CFR+ current strategy"
)
plt.xlabel("Iterations", fontweight="bold")
plt.ylabel("Exploitability (mbb/g)", fontweight="bold")
plt.loglog()
plt.legend()
plt.title("Comparison of current versus average strategy for Leduc poker", fontweight="bold")
plt.savefig(path_file.path
            + "project_plots_code/cfr(+)_regular_and_average_plot/"
            + "current_avg_comparison_leduc"
            + path_file.plot_type)
plt.show()

