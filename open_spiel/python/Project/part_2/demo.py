from open_spiel.python.project.part_2.algorithms.cfr import train_cfr
from open_spiel.python.project.part_2.algorithms.rcfr import train_rcfr
from open_spiel.python.project.part_2.calculate_store_plot_code import calculate_store_plot_cfr
from open_spiel.python.project.part_2.calculate_store_plot_code import calculate_store_plot_rcfr

# This is a demo file, used to showcase the functionality used when performing experiments.

########
# CFR+ #
########

# Example, training of CFR+
data = train_cfr(
    game_name="kuhn_poker",
    regret_matching_plus=True,
    alternating_updates=True,
    linear_averaging=True,
    average_policy=True, # use the current (false) or the average policy (true)
    iterations=int(1e1),
    print_freq=int(1e0),
    players=2,
    calculate_model=False, # do you want to create a csv-model?
    print_current_exploitability=False # for demo purposes, do not print the current exploitability
)
print("##### CFR+")
print("Saved info: ", data.keys())
print("Final exploitability", data["exploitability"][-1])
print("#####")

# Example, training + plotting the exploitability of CFR+
calculate_store_plot_cfr(
    game="leduc_poker",
    linear_averaging=True,
    alternating_updates=True,
    regret_matching_plus=True,
    average_policy=True,
    iterations=int(5e0),
    print_freq=int(1e0),
    print_current_exploitability=False
)
print("Plotting of CFR+ done")
print("#####\n")

##################
# Regression CFR #
##################

# Example, training Regression CFR
data = train_rcfr(
    game_name="kuhn_poker",
    eval_every=int(1e0),
    num_iterations=int(1e1),
    num_players=2,
    bootstrap=False, # use bootstrap or not
    buffer_size=-1,
    num_hidden_layers=2,
    num_hidden_units=6,
    num_epochs=10,
    batch_size=100,
    step_size=0.01,
    print_current_exploitability=False
)
print("##### Regression CFR+")
print("Saved info: ", data.keys())
print("Final exploitability ", data["exploitability"][-1])
print("#####")

# Example, training + plotting the exploitability of Regression CFR
calculate_store_plot_rcfr(
    game_name="kuhn_poker",
    eval_every=int(1e0),
    num_iterations=int(1e1),
    num_players=2,
    bootstrap=False,
    buffer_size=-1,
    num_hidden_layers=2,
    num_hidden_units=6,
    num_epochs=10,
    batch_size=100,
    step_size=0.01,
    print_current_exploitability=False
)
print("Plotting of RCFR done")
print("#####\n")