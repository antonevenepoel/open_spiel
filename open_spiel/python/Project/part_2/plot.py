
import numpy as np


from open_spiel.python.project.part_2.algorithms.fp import train_fp
from open_spiel.python.project.part_2.algorithms.nfsp import train_nfsp
from open_spiel.python.project.part_2.algorithms.cfr import train_cfr
from open_spiel.python.project.part_2.algorithms.deep_cfr import train_dcfr
from open_spiel.python.project.part_2.algorithms.deep_cfr_without_restart import train_dcfr_modified
from open_spiel.python.project.part_2.algorithms.rcfr import train_rcfr

import matplotlib.pyplot as plt
import numpy as np
from open_spiel.python.project.part_2 import path_file
from open_spiel.python.project.part_2.path_file import establish_path


def calculate_store_plot_cfr(
        game,
        regret_matching_plus,
        alternating_updates,
        linear_averaging,
        average_policy,
        print_freq,
        iterations,
        players=2
):
    print("##### CFR #####")
    output = train_cfr(
        game=game,
        players=players,
        print_freq=print_freq,
        iterations=iterations,
        average_policy=average_policy,
        regret_matching_plus=regret_matching_plus,
        alternating_updates=alternating_updates,
        linear_averaging=linear_averaging
    )

    # Save data in txt-file
    # Create meta data which is put in the header of the txt-files
    cfr_header = ""
    for key, value in output.items():
        if key not in ["iterations", "exploitability"]:
            cfr_header += f"{key}: {value} \n"
    # Iterations
    path = establish_path(path_file.data_path + "cfr_" + str(
        output["iterations"][-1] // int(1e3)) + "k_iterations_ITER_" + str(game), path_file.data_type)
    np.savetxt(
        fname=path,
        header=cfr_header,
        X=output["iterations"],
        delimiter=","
    )
    # Exploitability
    path = establish_path(path_file.data_path + "cfr_" + str(
        output["iterations"][-1] // int(1e3)) + "k_iterations_EXPL_" + str(game), path_file.data_type)
    np.savetxt(
        fname=path,
        header=cfr_header,
        X=output["exploitability"],
        delimiter=","
    )

    # Plots
    path = establish_path(path_file.plot_path
                          + "cfr_" + str(output["iterations"][-1] // int(1e3)) + "k_iterations_" + str(game), path_file.plot_type)
    plt.title("CFR: " + output["game"], fontweight="bold")
    plt.xlabel("Iterations", fontweight="bold")
    plt.ylabel("Exploitability", fontweight="bold")
    plt.plot(output["iterations"], output["exploitability"])
    plt.loglog()

    plt.savefig(path)
    plt.show()


def calculate_store_plot_dcfr(
        game_name,
        eval_every=int(1e2),
        num_iterations=int(1e3),
        num_traversals=40,
        policy_network_layers=(32, 32),
        advantage_network_layers=(16, 16),
        learning_rate=1e-3,
        batch_size_advantage=None,
        batch_size_strategy=None,
        memory_capacity=1e7
):
    print("##### DCFR #####")
    output = train_dcfr(
        game_name=game_name,
        eval_every=eval_every,
        num_iterations=num_iterations,
        num_traversals=num_traversals,
        policy_network_layers=policy_network_layers,
        advantage_network_layers=advantage_network_layers,
        learning_rate=learning_rate,
        batch_size_advantage=batch_size_advantage,
        batch_size_strategy=batch_size_strategy,
        memory_capacity=memory_capacity
    )

    # Save data in txt-file
    # Create meta data which is put in the header of the txt-files
    dcfr_header = ""
    for key, value in output.items():
        if key not in ["iterations", "exploitability"]:
            dcfr_header += f"{key}: {value} \n"
    # Iterations
    np.savetxt(
        fname=path_file.data_path + "dcfr_" + str(
            output["iterations"][-1]) + "_iterations_ITER" + path_file.data_type,
        header=dcfr_header,
        X=output["iterations"],
        delimiter=","
    )
    # Exploitability
    np.savetxt(
        fname=path_file.data_path + "dcfr_" + str(
            output["iterations"][-1]) + "_iterations_EXPL" + path_file.data_type,
        header=dcfr_header,
        X=output["exploitability"],
        delimiter=","
    )

    # Plots
    plt.title("DCFR: " + output["game"], fontweight="bold")
    plt.xlabel("Iterations", fontweight="bold")
    plt.ylabel("Exploitability", fontweight="bold")
    plt.plot(output["iterations"], output["exploitability"])
    plt.loglog()
    plt.savefig(path_file.plot_path
                + "dcfr_" + str(output["iterations"][-1]) + "_iterations"
                + path_file.plot_type,
                bbox_inches="tight")
    plt.show()


def calculate_store_plot_dcfr_modified(
        game_name,
        eval_every=int(1e2),
        num_iterations=int(1e3),
        num_traversals=40,
        policy_network_layers=(32, 32),
        advantage_network_layers=(16, 16),
        learning_rate=1e-3,
        batch_size_advantage=None,
        batch_size_strategy=None,
        memory_capacity=1e5
):
    print("##### DCFR_Modified #####")
    output = train_dcfr_modified(
        game_name=game_name,
        eval_every=eval_every,
        num_iterations=num_iterations,
        num_traversals=num_traversals,
        policy_network_layers=policy_network_layers,
        advantage_network_layers=advantage_network_layers,
        learning_rate=learning_rate,
        batch_size_advantage=batch_size_advantage,
        batch_size_strategy=batch_size_strategy,
        memory_capacity=memory_capacity
    )

    # Save data in txt-file
    # Create meta data which is put in the header of the txt-files
    dcfr_header = ""
    for key, value in output.items():
        if key not in ["iterations", "exploitability"]:
            dcfr_header += f"{key}: {value} \n"
    # Iterations
    path = establish_path(path_file.data_path + "dcfr_" + str(
        output["iterations"][-1]) + "_iterations_ITER", path_file.data_type)
    np.savetxt(
        fname=path,
        header=dcfr_header,
        X=output["iterations"],
        delimiter=","
    )
    # Exploitability
    path = establish_path(path_file.data_path + "dcfr_" + str(
        output["iterations"][-1]) + "_iterations_EXPL", path_file.data_type)
    np.savetxt(
        fname=path,
        header=dcfr_header,
        X=output["exploitability"],
        delimiter=","
    )

    # Plots
    intermediary_path = path_file.plot_path + "dcfr_" + str(output["iterations"][-1]) + "_iterations"
    path = establish_path(intermediary_path, path_file.plot_type)
    print(path)
    plt.title("DCFR: " + output["game"], fontweight="bold")
    plt.xlabel("Iterations", fontweight="bold")
    plt.ylabel("Exploitability", fontweight="bold")
    plt.plot(output["iterations"], output["exploitability"])
    plt.loglog()

    plt.savefig(path)

    plt.show()


def calculate_store_plot_fp(
        game="kuhn_poker",
        players=2,
        print_freq=int(1e2),
        iterations=int(1e3)
):
    print("##### FP #####")
    output = train_fp(
        game=game,
        players=players,
        print_freq=print_freq,
        iterations=iterations
    )
    #########################
    # Save data in txt-file #
    #########################
    # Create meta data which is put in the header of the txt-files
    fp_header = ""
    for key, value in output.items():
        if key not in ["iterations", "exploitability"]:
            fp_header += f"{key}: {value} \n"
    # Iterations
    np.savetxt(
        fname=path_file.data_path + "fp_" + str(
            output["iterations"][-1] // int(1e3)) + "k_iterations_ITER" + path_file.data_type,
        header=fp_header,
        X=output["iterations"],
        delimiter=","
    )
    # Exploitability
    np.savetxt(
        fname=path_file.data_path + "fp_" + str(
            output["iterations"][-1] // int(1e3)) + "k_iterations_EXPL" + path_file.data_type,
        header=fp_header,
        X=output["exploitability"],
        delimiter=","
    )
    ########
    # Plot #
    ########
    plt.title("FP: " + output["game"], fontweight="bold")
    plt.xlabel("Iterations", fontweight="bold")
    plt.ylabel("Exploitability", fontweight="bold")
    plt.plot(output["iterations"], output["exploitability"])
    plt.loglog()
    plt.savefig(path_file.plot_path
                + 'fp_' + str(output["iterations"][-1] // int(1e3)) + 'k_iterations'
                + path_file.plot_type,
                bbox_inches="tight")
    plt.show()


def calculate_store_plot_nfsp(
        game="kuhn_poker",
        num_players=2,
        hidden_layers_sizes=(64,),
        replay_buffer_capacity=int(2e5),
        num_train_episodes=int(1e4),
        epsilon_start=0.06,
        epsilon_end=0.001,
        reservoir_buffer_capacity=int(2e6),
        anticipatory_param=0.1,
        eval_every=int(1e4),
        batch_size=128,
        rl_learning_rate=0.01,
        sl_learning_rate=0.01,
        min_buffer_size_to_learn=1000,
        learn_every=64,
        optimizer_str="sgd"
):
    print("##### NFSP #####")
    output = train_nfsp(
        game=game,
        num_players=num_players,
        hidden_layers_sizes=hidden_layers_sizes,
        replay_buffer_capacity=replay_buffer_capacity,
        num_train_episodes=num_train_episodes,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        reservoir_buffer_capacity=reservoir_buffer_capacity,
        anticipatory_param=anticipatory_param,
        eval_every=eval_every,
        batch_size=batch_size,
        rl_learning_rate=rl_learning_rate,
        sl_learning_rate=sl_learning_rate,
        min_buffer_size_to_learn=min_buffer_size_to_learn,
        learn_every=learn_every,
        optimizer_str=optimizer_str
    )
    #########################
    # Save data in txt-file #
    #########################
    # Create meta data which is put in the header of the txt-files
    nfsp_header = ""
    for key, value in output.items():
        if key not in ["episodes", "exploitability", "losses"]:
            nfsp_header += f"{key}: {value} \n"
    # Episodes
    np.savetxt(
        fname=path_file.data_path + "nfsp_" + str(
            output["episodes"][-1] // int(1e3)) + "k_episodes_EPI" + path_file.data_type,
        header=nfsp_header,
        X=output["episodes"],
        delimiter=","
    )
    # Exploitability
    np.savetxt(
        fname=path_file.data_path + "nfsp_" + str(
            output["episodes"][-1] // int(1e3)) + "k_episodes_EXPL" + path_file.data_type,
        header=nfsp_header,
        X=output["exploitability"],
        delimiter=",")
    # TODO: Losses - Misschien nog interessant om te bekijken later
    # np.savetxt(
    #     fname=paths.data_path + "nfsp_" + str(output["episodes"][-1]//int(1e3)) + "k_episodes_LOSS" + paths.data_type,
    #     header="",
    #     X=output["losses"],
    #     delimiter=","
    # )

    ########
    # Plot #
    ########
    plt.title("NFSP: " + output["game"], fontweight="bold")
    plt.xlabel("Episodes", fontweight="bold")
    plt.ylabel("Exploitability", fontweight="bold")
    plt.plot(output["episodes"], output["exploitability"])
    plt.loglog()
    plt.savefig(path_file.plot_path
                + "nfsp_" + str(output["episodes"][-1] // int(1e3)) + "k_episodes"
                + path_file.plot_type,
                bbox_inches="tight")
    plt.show()




def calculate_store_plot_rcfr(
        game_name="kuhn_poker",
        eval_every=int(1e2),
        num_iterations=int(1e3),
        num_players=2,
        bootstrap=False,
        truncate_negative=False,
        buffer_size=-1,
        num_hidden_layers=1,
        num_hidden_units=13,
        num_hidden_factors=8,
        use_skip_connections=True,
        num_epochs=200,
        batch_size=100,
        step_size=0.01
):
    print("##### RCFR #####")
    output = train_rcfr(
        game_name=game_name,
        eval_every=eval_every,
        num_iterations=num_iterations,
        num_players=num_players,
        bootstrap=bootstrap,
        truncate_negative=truncate_negative,
        buffer_size=buffer_size,
        num_hidden_layers=num_hidden_layers,
        num_hidden_units=num_hidden_units,
        num_hidden_factors=num_hidden_factors,
        use_skip_connections=use_skip_connections,
        num_epochs=num_epochs,
        batch_size=batch_size,
        step_size=step_size
    )
    #########################
    # Save data in txt-file #
    #########################
    # Create meta data which is put in the header of the txt-files
    rcfr_header = ""
    for key, value in output.items():
        if key not in ["iterations", "exploitability"]:
            rcfr_header += f"{key}: {value} \n"
    # Iterations
    path = establish_path(path_file.data_path + "rcfr_" + str(
            output["iterations"][-1]) + "_iterations_ITER",  path_file.data_type)
    np.savetxt(
        fname= path ,
        header=rcfr_header,
        X=output["iterations"],
        delimiter=","
    )
    # Exploitability
    path= establish_path(path_file.data_path + "rcfr_" + str(
            output["iterations"][-1]) + "_iterations_EXPL",  path_file.data_type)
    np.savetxt(
        fname=path,
        header=rcfr_header,
        X=output["exploitability"],
        delimiter=","
    )

    # Plots
    intermediary_path = path_file.plot_path + "rcfr_" + str(output["iterations"][-1]) + "_iterations"
    path = establish_path(intermediary_path, path_file.plot_type)
    print(path)
    plt.title("RCFR: " + output["game"], fontweight="bold")
    plt.xlabel("Iterations", fontweight="bold")
    plt.ylabel("Exploitability", fontweight="bold")
    plt.plot(output["iterations"], output["exploitability"])
    plt.loglog()

    plt.savefig(path)

    plt.show()


if __name__ == "__main__":
    # calculate_store_plot_dcfr_modified(
    #     game_name="kuhn_poker",
    #     eval_every=int(1e1),
    #     num_iterations=int(1000),
    #     num_traversals=1000,
    #     policy_network_layers=(16,16),
    #     advantage_network_layers=(16,16),
    #     learning_rate=1e-2,
    #     memory_capacity=1e7,
    #     batch_size_advantage=None,
    #     batch_size_strategy=None
    # )

    calculate_store_plot_rcfr(
        game_name="leduc_poker",
        eval_every= int(1),
        num_iterations=int(3 ),
        num_hidden_layers=int(2),
        num_hidden_units=int(100),
        num_hidden_factors=int(0),
    )

    # calculate_store_plot_cfr(
    #     game ="leduc_poker",
    #     players=2,
    #     print_freq=int(1e1),
    #     iterations=int(5000)
    # )
    # calculate_store_plot_fp(
    #     game ="leduc_poker",
    #     players=2,
    #     print_freq=int(1e1),
    #     iterations=int(1e3)
    # )

    # calculate_store_plot_cfr(
    #     game="leduc_poker",
    #     linear_averaging=True,
    #     alternating_updates=True,
    #     regret_matching_plus=True,
    #     average_policy=True,
    #     print_freq=1,
    #     iterations=int(1e1)
    # )

