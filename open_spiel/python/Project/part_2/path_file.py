import os

PATH_FLAG = False  # zet hier op True als je jouw path wilt gebruiken
PATH_GUEST = False # zet hier op True als je PATH_GUEST wilt gebruiken

# PATHS
path_guest = "..."
path_arnout = "/Users/arnouthillen/open_spiel/open_spiel/python/project/part_2/"
path_anton = "/Users/antonevenepoel/Github/open_spiel2/open_spiel/python/project/part_2/rcfr-allin"  # invullen

# Path selection
if PATH_GUEST:
    path = StopAsyncIteration
else:
    path = path_arnout if not PATH_FLAG else path_anton

# Store paths
plot_path = path + "plots/"
plot_type = ".png"

data_path = path + "data_arnout/"
data_type = ".txt"

# Dynamic paths
def establish_path(path, type):
    i = 0
    if os.path.isfile(path + type):
        path = path + str(i)
        i += 1
    while (os.path.isfile(path + type)):
        path = path[:len(path) - 1] + str(i)
        i += 1
    return path + type
