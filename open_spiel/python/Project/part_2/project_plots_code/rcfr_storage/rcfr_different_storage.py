import matplotlib.pyplot as plt
import numpy as np
from open_spiel.python.project.part_2 import path_file

########
# PATH #
########
path = "data/"
path1 = path + "rcfr_1000_iterations_ITER"
path2 = path + "rcfr_1000_iterations_EXPL"

########
# PLOT #
########
plt.plot(
    np.loadtxt(path1 + "0.txt"),
    [i*10**3 for i in np.loadtxt(path2 + "0.txt")],
    label="infinite data storage"
)
plt.plot(
    np.loadtxt(path1 + "1.txt"),
    [i*10**3 for i in np.loadtxt(path2 + "1.txt")],
    label="bootstrap"
)
plt.plot(
    np.loadtxt(path1 + "2.txt"),
    [i*10**3 for i in np.loadtxt(path2 + "2.txt")],
    label="reservoir buffer"
)
plt.xlabel("Iterations", fontweight="bold")
plt.ylabel("Exploitability (mbb/g)", fontweight="bold")
plt.loglog()
plt.legend()
plt.title("RCFR in Leduc poker: Storage", fontweight="bold")
plt.savefig("rcfr_storage_leduc"
            + path_file.plot_type,
            bbox_inches="tight")
plt.show()
