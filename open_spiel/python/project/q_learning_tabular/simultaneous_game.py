# Copyright 2019 DeepMind Technologies Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tabular Q-Learner example on Tic Tac Toe.

Two Q-Learning agents are trained by playing against each other. Then, the game
can be played against the agents from the command line.

After about 10**5 training episodes, the agents reach a good policy: win rate
against random opponents is around 99% for player 0 and 92% for player 1.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import sys
from absl import app
from absl import flags
import numpy as np
from six.moves import input
from six.moves import range

import pyspiel

from open_spiel.python import rl_environment
from open_spiel.python.algorithms import random_agent
from open_spiel.python.algorithms import tabular_qlearner

FLAGS = flags.FLAGS

flags.DEFINE_integer("num_episodes", int(1e5), "Number of train episodes.")


def eval_against_random_bots(env, trained_agents, random_agents, num_episodes, game_number):
    """Evaluates `trained_agents` against `random_agents` for `num_episodes`."""
    state_count_0, state_count_1 = np.zeros(4), np.zeros(4)
    state_count_rps_0, state_count_rps_1 = np.zeros(9), np.zeros(9)
    for player_pos in range(2):
        if player_pos == 0:
            cur_agents = [trained_agents[0], random_agents[1]]
        else:
            cur_agents = [random_agents[0], trained_agents[1]]
        for _ in range(num_episodes):
            time_step = env.reset()
            while not time_step.last():
                action0 = cur_agents[0].step(time_step, is_evaluation=True).action
                action1 = cur_agents[1].step(time_step, is_evaluation=True).action
                time_step = env.step([action0, action1])
            # prisoner's dilemma
            if game_number == 1:
                for i, state in enumerate([[3, 3], [0, 5], [5, 0], [1, 1]]):
                    if state == time_step.rewards:
                        # Hoog state_count_0 of state_count_1 op op basis van player_pos
                        locals()["state_count_{}".format(player_pos)][i] += 1
            # matching pennies
            elif game_number == 2:
                for i, state in enumerate([[1, -1], [-1, 1], [-1, 1], [1, -1]]):
                    if state == time_step.rewards:
                        # Hoog state_count_0 of state_count_1 op op basis van player_pos
                        locals()["state_count_{}".format(player_pos)][i] += 1
            # battle of the sexes
            elif game_number == 3:
                for i, state in enumerate([[2, 1], [0, 0], [1, 2], [0, 0]]):
                    if state == time_step.rewards:
                        # Hoog state_count_0 of state_count_1 op op basis van player_pos
                        locals()["state_count_{}".format(player_pos)][i] += 1
            # rock, paper, scissors
            elif game_number == 4:
                for i, state in enumerate([[0, 0], [-1, 1], [1, -1], [1, -1], [0, 0], [-1, 1], [-1, 1], [1, -1], [0, 0]]):
                    if state == time_step.rewards:
                        # Hoog state_count_0 of state_count_1 op op basis van player_pos
                        locals()["state_count_rps_{}".format(player_pos)][i] += 1
    if game_number != 4:
        return state_count_0 / num_episodes, state_count_1 / num_episodes
    else:
        return state_count_rps_0 / num_episodes, state_count_rps_1 / num_episodes


def main(_):
    game1 = pyspiel.create_matrix_game("prisoners_dilemma", "Prisoner's Dilemma",
                                      ["C", "B"], ["C", "B"],
                                      [[3, 0], [5, 1]], [[3, 5], [0, 1]])
    game2 = pyspiel.create_matrix_game("matching_pennies", "Matching Pennies",
                                      ["H", "T"], ["H", "T"],
                                      [[1, -1], [-1, 1]], [[-1, 1], [1, -1]])
    game3 = pyspiel.create_matrix_game("battle_of_the_sexes", "Battle of the Sexes",
                                      ["B", "S"], ["B", "S"],
                                      [[2, 0], [0, 1]], [[1, 0], [0, 2]])
    game4 = pyspiel.create_matrix_game("rock_paper_scissors", "Rock, Paper, Scissors",
                                      ["R", "P", "S"], ["R", "P", "S"],
                                      [[0, -1, 1], [1, 0, -1], [-1, 1, 0]], [[0, 1, -1],[-1, 0, 1], [1, 0, -1]])
    num_players = 2

    game_number = input("Choose a game\n - Prisoner's Dilemma: 1 \n - Matching Pennies: 2 \n - Battle of the Sexes: 3 \n - Rock, Paper, Scissors: 4 \n\n Game number is ")
    game = locals()["game{}".format(game_number)]

    env = rl_environment.Environment(game)
    num_actions = env.action_spec()["num_actions"]

    agents = [
        tabular_qlearner.QLearner(player_id=idx, num_actions=num_actions)
        for idx in range(num_players)
    ]

    # random agents for evaluation
    random_agents = [
        random_agent.RandomAgent(player_id=idx, num_actions=num_actions)
        for idx in range(num_players)
    ]

    # 1. Train the agents
    training_episodes = FLAGS.num_episodes
    for cur_episode in range(training_episodes):
        if cur_episode % int(1e4) == 0:
            state_rate_0, state_rate_1 = eval_against_random_bots(env, agents, random_agents, 1000, int(game_number))
            logging.info("Starting episode %s, state_rate P0 %s, state_rate P1 %s", cur_episode, state_rate_0, state_rate_1)
        time_step = env.reset()
        while not time_step.last():
            action0 = agents[0].step(time_step).action
            action1 = agents[1].step(time_step).action
            time_step = env.step([action0, action1])
        # Episode is over, step all agents with final info state.
        for agent in agents:
            agent.step(time_step)

# De output stelt kans dat speler 0 en 1 voor één van de states kiest, en dit tegen een tegenstander die random speelt.
# 1. Voor prisoner's dilemma zie je dat speler 0 kiest voor states 3 en 4 (wat neerkomt op verraadt); speler 1 kiest voor states 2 en 4 (ook verraadt).
# 2. Voor Matching Pennies kiezen beide spelers alle vier de states uniform.
# 3.
# 4.

if __name__ == "__main__":
  app.run(main)
