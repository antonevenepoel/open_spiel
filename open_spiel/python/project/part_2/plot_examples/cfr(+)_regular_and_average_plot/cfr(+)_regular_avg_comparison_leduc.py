import matplotlib.pyplot as plt
import numpy as np

from open_spiel.python.project.part_2 import path_file

plt.plot(
    np.loadtxt("data/cfr_10k_iterations_ITER0_avg.txt"),
    [i*10**3 for i in np.loadtxt("data/cfr_10k_iterations_EXPL0_avg.txt")],
    label="CFR average strategy"
)
plt.plot(
    np.loadtxt("data/cfr_10k_iterations_ITER0_avg.txt"),
    [i*10**3 for i in np.loadtxt("data/cfr_10k_iterations_EXPL0_avg.txt")],
    label="CFR current strategy"
)
plt.plot(
    np.loadtxt("data/cfr_10k_iterations_ITER0_avg.txt"),
    [i*10**3 for i in np.loadtxt("data/cfr_10k_iterations_EXPL0_avg.txt")],
    label="CFR+ average strategy"
)
plt.plot(
    np.loadtxt("data/cfr_10k_iterations_ITER0_avg.txt"),
    [i*10**3 for i in np.loadtxt("data/cfr_10k_iterations_EXPL0_avg.txt")],
    label="CFR+ current strategy"
)
plt.xlabel("Iterations", fontweight="bold")
plt.ylabel("Exploitability (mbb/g)", fontweight="bold")
plt.loglog()
plt.title("Comparison of current versus average strategy for Leduc poker", fontweight="bold")
plt.savefig(path_file.path
            + "plot_examples/cfr(+)_regular_and_average_plot/"
            + "current_avg_comparison_leduc"
            + path_file.plot_type)
plt.show()

