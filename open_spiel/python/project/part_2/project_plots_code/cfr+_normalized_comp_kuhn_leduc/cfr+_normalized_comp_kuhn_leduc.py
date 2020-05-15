import matplotlib.pyplot as plt
import numpy as np

from open_spiel.python.project.part_2 import path_file

plt.plot(
    np.loadtxt("data/cfr_10k_iterations_ITER_kuhn_poker1.txt"),
    [i*10**3/12 for i in np.loadtxt("data/cfr_10k_iterations_EXPL_kuhn_poker1.txt")],
    label="Kuhn"
)
plt.plot(
    np.loadtxt("data/cfr_10k_iterations_ITER7.txt"),
    [i*10**3/936 for i in np.loadtxt("data/cfr_10k_iterations_EXPL7.txt")],
    label="Leduc"
)
plt.xlabel("Iterations", fontweight="bold")
plt.ylabel("Exploitability/# information sets (mbb/g)", fontweight="bold")
plt.loglog()
plt.legend()
plt.title("Comparison of normalized CFR+ in Kuhn and Leduc", fontweight="bold")
plt.savefig(path_file.path
            + "project_plots_code/cfr+_normalized_comp_kuhn_leduc/"
            + "cfr+_normalized_comp_kuhn_leduc"
            + path_file.plot_type,)
plt.show()
