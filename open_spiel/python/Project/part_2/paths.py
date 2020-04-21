PATH_FLAG = True # zet hier op True als je jouw pad wilt gebruiken

path_arnout = "/Users/arnouthillen/open_spiel/open_spiel/python/project/part_2"
path_anton = "/Users/antonevenepoel/Github/open_spiel2/open_spiel/python/project/part_2" # invullen

path = path_arnout if not PATH_FLAG else path_anton

plot_path = path + "/plots/"
plot_type = ".png"

data_path = path + "/data/"
data_type = ".txt"
