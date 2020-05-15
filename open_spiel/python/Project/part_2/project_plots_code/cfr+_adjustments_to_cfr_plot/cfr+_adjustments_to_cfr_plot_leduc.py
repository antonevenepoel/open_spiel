import matplotlib.pyplot as plt
import numpy as np

from open_spiel.python.project.part_2 import path_file

plt.plot(
    np.loadtxt("leduc_data/cfr_1k_iterations_ITER0.txt"),
    [i*10**3 for i in np.loadtxt("leduc_data/cfr_1k_iterations_EXPL0.txt")],
    label="CFR"
)
plt.plot(
    np.loadtxt("leduc_data/cfr_1k_iterations_ITER3.txt"),
    [i*10**3 for i in np.loadtxt("leduc_data/cfr_1k_iterations_EXPL3.txt")],
    label="CFR and alternating updates"
)
plt.plot(
    np.loadtxt("leduc_data/cfr_1k_iterations_ITER1.txt"),
    [i*10**3 for i in np.loadtxt("leduc_data/cfr_1k_iterations_EXPL1.txt")],
    label="CFR and linear averaging"
)
plt.plot(
    np.loadtxt("leduc_data/cfr_1k_iterations_ITER2.txt"),
    [i*10**3 for i in np.loadtxt("leduc_data/cfr_1k_iterations_EXPL2.txt")],
    label="CFR and regret matching+"
)
plt.plot(
    np.loadtxt("leduc_data/cfr_1k_iterations_ITER4.txt"),
    [i*10**3 for i in np.loadtxt("leduc_data/cfr_1k_iterations_EXPL4.txt")],
    label="CFR+"
)
plt.xlabel("Iterations", fontweight="bold")
plt.ylabel("Exploitability (mbb/g)", fontweight="bold")
plt.loglog()
plt.legend()
plt.title("Comparison of individual CFR+ adjustments to CFR in Leduc poker", fontweight="bold")
plt.savefig(path_file.path
            + "project_plots_code/cfr+_adjustments_to_cfr_plot/"
            + "cfr+_adjustments_leduc"
            + path_file.plot_type,
            bbox_inches="tight")
plt.show()

