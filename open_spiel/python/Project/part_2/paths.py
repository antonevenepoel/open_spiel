PATH_FLAG = False # zet hier op True als je jouw pad wilt gebruiken

path_arnout = "/Users/arnouthillen/open_spiel/open_spiel/python/project/part_2"
path_anton = "..." # invullen

path = path_arnout if not PATH_FLAG else path_anton

plot_path = path + "/plots/"
plot_type = ".png"

data_path = path + "/data/"
data_type = ".txt"
