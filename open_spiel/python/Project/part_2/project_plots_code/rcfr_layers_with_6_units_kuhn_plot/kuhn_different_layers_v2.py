import matplotlib.pyplot as plt
import numpy as np
from open_spiel.python.project.part_2 import path_file

########
# PATH #
########
path = "data/"
path1 = path + "rcfr_1000_iterations_ITER"
path2 = path +"rcfr_1000_iterations_EXPL"

########
# PLOT #
########
plt.plot(
     np.loadtxt(path1 + ".txt"),
     [i*10**3 for i in np.loadtxt(path2 + ".txt")],
     label="hl: 0"
)
plt.plot(
     np.loadtxt(path1 + "0.txt"),
     [i*10**3 for i in np.loadtxt(path2 + "0.txt")],
     label="hl: 1"
)
plt.plot(
     np.loadtxt(path1 + "1.txt"),
     [i*10**3 for i in np.loadtxt(path2 + "1.txt")],
     label="hl: 2"
)
plt.plot(
    np.loadtxt(path1 + "2.txt"),
    [i*10**3 for i in np.loadtxt(path2 + "2.txt")],
    label="hl: 3"
)
plt.plot(
    np.loadtxt(path1 + "3.txt"),
    [i*10**3 for i in np.loadtxt(path2 + "3.txt")],
    label="hl: 4"
)
plt.xlabel("Iterations", fontweight="bold")
plt.ylabel("Exploitability (mbb/g)", fontweight="bold")
plt.loglog()
plt.legend()
plt.title("RCFR in Kuhn poker: Hidden Layers", fontweight="bold")
plt.savefig("rcfr_hiddenlayers_kuhn"
            + path_file.plot_type,
            bbox_inches="tight")
plt.show()
