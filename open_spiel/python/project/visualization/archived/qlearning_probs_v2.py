import pyspiel

import numpy as np
from open_spiel.python import rl_environment
from open_spiel.python.algorithms import tabular_qlearner, random_agent
from open_spiel.python.Project.q_learning_tabular.simultaneous_game import eval_two_agents

def calculate_probs(game_name):
    game_dict = {
        "prisoner's dilemma": 1,
        "matching pennies": 2,
        "battle of the sexes": 3,
        "biased rock, paper, scissors": 4
    }
    game_number = game_dict[game_name]
    assert game_number in {1,2,3,4}

    game1 = pyspiel.create_matrix_game("prisoners_dilemma", "Prisoner's Dilemma",
                                          ["C", "B"], ["C", "B"],
                                          [[3, 0], [5, 1]], [[3, 5], [0, 1]])
    game2 = pyspiel.create_matrix_game("matching_pennies", "Matching Pennies",
                                      ["H", "T"], ["H", "T"],
                                      [[1, -1], [-1, 1]], [[-1, 1], [1, -1]])
    game3 = pyspiel.create_matrix_game("battle_of_the_sexes", "Battle of the Sexes",
                                      ["B", "S"], ["B", "S"],
                                      [[2, 0], [0, 1]], [[1, 0], [0, 2]])
    game4 = pyspiel.create_matrix_game("biased_rock_paper_scissors", "Biased Rock, Paper, Scissors",
                                      ["R", "P", "S"], ["R", "P", "S"],
                                      [[0, -0.25, 0.5], [0.25, 0, -0.05], [-0.5, 0.05, 0]], [[0, 0.25, -0.5],[-0.25, 0, 0.05], [0.5, -0.05, 0]])
    game = locals()["game{}".format(game_number)]
    env0, env1 = rl_environment.Environment(game), rl_environment.Environment(game)
    num_actions = env0.action_spec()["num_actions"]
    num_players = 2

    agents = [
        tabular_qlearner.QLearner(player_id=idx, num_actions=num_actions)
        for idx in range(num_players)
    ]
    random_agents = [
        random_agent.RandomAgent(player_id=idx, num_actions=num_actions)
        for idx in range(num_players)
    ]

    # 1. Train the agents
    training_episodes = int(10e3)
    probs = np.zeros(shape=(training_episodes, 2)) if game_name != "biased rock, paper, scissors" else np.zeros(shape=(training_episodes,3))
    for cur_episode in range(training_episodes):
        # if cur_episode % int(1e4) == 0:
        #     state_rate_self_play = eval_two_agents(env, agents[0], agents[1], 10000, int(game_number))
        #     state_rate_random = eval_two_agents(env, agents[0], random_agents[1], 10000, int(game_number))
        #     logging.info("Starting episode %s, state_rate_self_play %s, state_rate_random %s", cur_episode,
        #                  state_rate_self_play, state_rate_random)
        time_step0, time_step1 = env0.reset(), env1.reset()
        while not time_step0.last():
            step_output0 = agents[0].step(time_step0)
            step_output1 = random_agents[1].step(time_step0)
            if game_name != "biased rock, paper, scissors":
                probs[cur_episode][0] = step_output0.probs[0]
            else:
                probs[cur_episode] = step_output0.probs
            action0 = step_output0.action
            action1 = step_output1.action
            time_step0 = env0.step([action0, action1])
        # Episode is over, step all agents with final info state.
        for agent in [agents[0], random_agents[1]]:
            agent.step(time_step0)
            

        while not time_step1.last():
            step_output0 = random_agents[0].step(time_step1)
            step_output1 = agents[1].step(time_step1)
            if game_name != "biased rock, paper, scissors":
                probs[cur_episode][1] = step_output1.probs[0]
            action0 = step_output0.action
            action1 = step_output1.action
            time_step1 = env1.step([action0, action1])
        # Episode is over, step all agents with final info state.
        for agent in [random_agents[0], agents[1]]:
            agent.step(time_step1)
    return probs


if __name__ == '__main__':
    None