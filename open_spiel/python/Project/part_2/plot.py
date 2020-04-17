from open_spiel.python.project.part_2.fp import train_fp
from open_spiel.python.project.part_2.nfsp import train_nfsp
from open_spiel.python.project.part_2.cfr import train_cfr
from open_spiel.python.project.part_2.dcfr import train_dcfr

import matplotlib.pyplot as plt
from open_spiel.python.project.part_2 import paths

path = paths.path_arnout # aanpassen naar path_anton (zie `paths.py`)

output_fp = train_fp(
    game="kuhn_poker",
    number_of_players=2,
    print_freq=int(1e3),
    iterations=int(1e4)
)
output_nfsp = train_nfsp(
    game="kuhn_poker",
    num_players=2,
    hidden_layers_sizes=(64,),
    replay_buffer_capacity=int(2e5),
    num_train_episodes=int(1e4),
    epsilon_start=0.06,
    epsilon_end=0.001,
    reservoir_buffer_capacity=int(2e6),
    anticipatory_param=0.1,
    eval_every=int(1e4)
)
output_cfr = train_cfr(
    game="kuhn_poker",
    players=2,
    iterations=int(1e4),
    print_freq=int(1e3)
)
# werkt nog niet
output_dcfr = train_dcfr(
    game_name="kuhn_poker",
    num_iterations=int(1e2),
    num_traversals=40,
    policy_network_layers=(32, 32),
    advantage_network_layers=(16, 16),
    learning_rate=1e-3,
    batch_size_advantage=None,
    batch_size_strategy=None,
    memory_capacity=1e7
)

# plots
plt.title("NFSP: " + output["game"], fontweight="bold")
plt.xlabel("Episodes", fontweight="bold")
plt.ylabel("Exploitability", fontweight="bold")
plt.plot(output["episodes"], output["exploitability"])
plt.loglog()
plt.savefig(paths.path_arnout
            + 'nfsp_' + str(output["episodes"][-1]) + '_episodes'
            + '.' + paths.type)
plt.show()