import matplotlib.pyplot as plt
import numpy as np

from open_spiel.python.project.part_2 import path_file

plt.plot(
    np.loadtxt("kuhn_data/cfr_10k_iterations_ITER8.txt"),
    [i*10**3 for i in np.loadtxt("kuhn_data/cfr_10k_iterations_EXPL8.txt")],
    label="CFR"
)
plt.plot(
    np.loadtxt("kuhn_data/cfr_10k_iterations_ITER2.txt"),
    [i*10**3 for i in np.loadtxt("kuhn_data/cfr_10k_iterations_EXPL2.txt")],
    label="CFR and alternating updates"
)
plt.plot(
    np.loadtxt("kuhn_data/cfr_10k_iterations_ITER4.txt"),
    [i*10**3 for i in np.loadtxt("kuhn_data/cfr_10k_iterations_EXPL4.txt")],
    label="CFR and linear averaging"
)
plt.plot(
    np.loadtxt("kuhn_data/cfr_10k_iterations_ITER3.txt"),
    [i*10**3 for i in np.loadtxt("kuhn_data/cfr_10k_iterations_EXPL3.txt")],
    label="CFR and regret matching+"
)
plt.plot(
    np.loadtxt("kuhn_data/cfr_10k_iterations_ITER6.txt"),
    [i*10**3 for i in np.loadtxt("kuhn_data/cfr_10k_iterations_EXPL6.txt")],
    label="CFR+"
)
plt.xlabel("Iterations", fontweight="bold")
plt.ylabel("Exploitability (mbb/g)", fontweight="bold")
plt.loglog()
plt.legend()
plt.title("Comparison of individual CFR+ adjustments to CFR in Kuhn poker", fontweight="bold")
plt.savefig(path_file.path
            + "project_plots_code/cfr+_adjustments_to_cfr_plot/"
            + "cfr+_adjustments_kuhn"
            + path_file.plot_type,
            bbox_inches="tight")
plt.show()
