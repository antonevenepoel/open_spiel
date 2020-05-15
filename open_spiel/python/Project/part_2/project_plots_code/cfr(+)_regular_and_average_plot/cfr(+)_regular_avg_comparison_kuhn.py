import matplotlib.pyplot as plt
import numpy as np

from open_spiel.python.project.part_2 import path_file

plt.plot(
    np.loadtxt("kuhn_data/cfr_10k_iterations_ITER8_avg.txt"),
    [i*10**3 for i in np.loadtxt("kuhn_data/cfr_10k_iterations_EXPL8_avg.txt")],
    label="CFR average strategy"
)
plt.plot(
    np.loadtxt("kuhn_data/cfr_10k_iterations_ITER_kuhn_poker.txt"),
    [i*10**3 for i in np.loadtxt("kuhn_data/cfr_10k_iterations_EXPL_kuhn_poker.txt")],
    label="CFR current strategy"
)
plt.plot(
    np.loadtxt("kuhn_data/cfr_10k_iterations_ITER6_avg.txt"),
    [i*10**3 for i in np.loadtxt("kuhn_data/cfr_10k_iterations_EXPL6_avg.txt")],
    label="CFR+ average strategy"
)
plt.plot(
    np.loadtxt("kuhn_data/cfr_10k_iterations_ITER_kuhn_poker0.txt"),
    [i*10**3 for i in np.loadtxt("kuhn_data/cfr_10k_iterations_EXPL_kuhn_poker0.txt")],
    label="CFR+ current strategy"
)
plt.xlabel("Iterations", fontweight="bold")
plt.ylabel("Exploitability (mbb/g)", fontweight="bold")
plt.legend()
plt.loglog()
plt.title("Comparison of current versus average strategy for Kuhn poker", fontweight="bold")
plt.savefig(path_file.path
            + "project_plots_code/cfr(+)_regular_and_average_plot/"
            + "current_avg_comparison_kuhn"
            + path_file.plot_type)
plt.show()

