import matplotlib.pyplot as plt
import numpy as np

from open_spiel.python.project.part_2 import path_file
from open_spiel.python.project.part_2.plot import calculate_store_plot_cfr

###########
# EXAMPLE #
###########
path = "units/"
path1 = path + "rcfr_1000_iterations_ITER"
path2 = path +"rcfr_1000_iterations_EXPL"

plt.plot(
    np.loadtxt(path1 + "4.txt"),
    [i*10**3 for i in np.loadtxt(path2 + "4.txt")],
    label="hu: 4"
)

plt.plot(
     np.loadtxt(path1 + "3.txt"),
     [i*10**3 for i in np.loadtxt(path2 + "3.txt")],
     label="hu: 5" )

plt.plot(
     np.loadtxt(path1 + ".txt"),
     [i*10**3 for i in np.loadtxt(path2 + ".txt")],
     label="hu: 6" )

plt.plot(
     np.loadtxt(path1 + "0.txt"),
     [i*10**3 for i in np.loadtxt(path2 + "0.txt")],
     label="hu: 7" )


plt.plot(
    np.loadtxt(path1 + "1.txt"),
    [i*10**3 for i in np.loadtxt(path2 + "1.txt")],
    label="hu: 8"
)
# plt.plot(
#      np.loadtxt(path1 + "2.txt"),
#      [i*10**3 for i in np.loadtxt(path2 + "2.txt")],
#      label="hu: 9" )
# plt.plot(
#     np.loadtxt(path1 + "0.txt"),
#     np.loadtxt(path2 + "0.txt"),
#     label="hu: 30"
# )
# plt.plot(
#     np.loadtxt(path1 + "1.txt"),
#     np.loadtxt(path2 + "1.txt"),
#     label="hu: 50"
# )
# plt.plot(
#     np.loadtxt(path1 + "2.txt"),
#     np.loadtxt(path2 + "2.txt"),
#     label="hu: 100"
# )


plt.xlabel("Iterations", fontweight="bold")
plt.ylabel("Exploitability (mbb/g)", fontweight="bold")
plt.loglog()
plt.legend()
plt.title("RCFR in Kuhn poker: Hidden Units", fontweight="bold")
plt.savefig(path
            + "rcfr_hiddenunits_kuhn"
            + path_file.plot_type,
            bbox_inches="tight")
plt.show()
