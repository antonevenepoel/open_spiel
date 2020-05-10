import matplotlib.pyplot as plt
import numpy as np

from open_spiel.python.project.part_2 import path_file

plt.plot(
    np.loadtxt("data/cfr_10k_iterations_ITER_leduc_poker.txt"),
    [i*10**3 for i in np.loadtxt("data/cfr_10k_iterations_EXPL_leduc_poker.txt")],
    label="CFR"
)
plt.xlabel("Iterations", fontweight="bold")
plt.ylabel("Exploitability (mbb/g)", fontweight="bold")
plt.loglog()
plt.title("Exploitability of CFR-agent for Leduc poker", fontweight="bold")
plt.savefig(path_file.path
            + "plot_examples/regular_cfr_plot/"
            + "cfr_leduc"
            + path_file.plot_type)
plt.show()