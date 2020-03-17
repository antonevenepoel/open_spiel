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


def eval_two_agents(env, agent0, agent1, num_episodes, game_number):
    """Evaluates `trained_agents` against `random_agents` for `num_episodes`."""
    state_count = np.zeros(4)
    state_count_rps = np.zeros(9)
    reward_count = 0
    for _ in range(num_episodes):
        time_step = env.reset()
        while not time_step.last():
            action0 = agent0.step(time_step, is_evaluation=True).action
            action1 = agent1.step(time_step, is_evaluation=True).action
            time_step = env.step([action0, action1])
        # prisoner's dilemma
        if game_number == 1:
            for i, state in enumerate([[3, 3], [0, 5], [5, 0], [1, 1]]):
                if state == time_step.rewards:
                    state_count[i] += 1
                    reward_count += state[0]
        # matching pennies
        elif game_number == 2:
            reward = time_step.rewards
            if reward == [1, -1]:
                # Linksboven
                if action0 == 0:
                    state_count[0] += 1
                # Rechtsonder
                else:
                    state_count[3] += 1
            elif reward == [-1, 1]:
                # Rechtsboven
                if action0 == 0:
                    state_count[1] += 1
                # Linksonder
                else:
                    state_count[2] += 1
            reward_count += reward[0]
        # battle of the sexes
        elif game_number == 3:
            for i, state in enumerate([[2, 1], [0, 0], [0, 0], [1, 2]]):
                if state == time_step.rewards:
                    # de `if` vermijdt dat er wordt geteld tijdens foute combinaties (vermijdt dubbeltellen).
                    state_count[i] += 1 if not (i == 1 and action0 == 1 or i == 2 and action0 == 0) else 0
                    reward_count += state[0] if not (i == 1 and action0 == 1 or i == 2 and action0 == 0) else 0
        # rock, paper, scissors
        elif game_number == 4:
            for i, state in enumerate([[0, 0], [-0.25, 0.25], [0.5, -0.5], [0.25, -0.25], [0, 0], [-0.05, 0.05], [-0.5, 0.5], [0.05, -0.05], [0, 0]]):
                if state == time_step.rewards:
                    state_count_rps[i] += 1 if not (i == 0 and (action0 == 1 or action0 == 2)
                                                    or i == 4 and (action0 == 0 or action0 == 2)
                                                    or i == 8 and (action0 == 0 or action0 == 1)) else 0
                    reward_count += state[0] if not (i == 0 and (action0 == 1 or action0 == 2)
                                                    or i == 4 and (action0 == 0 or action0 == 2)
                                                    or i == 8 and (action0 == 0 or action0 == 1)) else 0
    logging.info("Average utility for player 0: " + str(reward_count/num_episodes))
    if game_number != 4:
        return state_count / num_episodes
    else:
        return state_count_rps / num_episodes


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
    game4 = pyspiel.create_matrix_game("biased_rock_paper_scissors", "Biased Rock, Paper, Scissors",
                                      ["R", "P", "S"], ["R", "P", "S"],
                                      [[0, -0.25, 0.5], [0.25, 0, -0.05], [-0.5, 0.05, 0]], [[0, 0.25, -0.5],[-0.25, 0, 0.05], [0.5, -0.05, 0]])

    game_number = input("Choose a game\n - Prisoner's Dilemma: 1 \n - Matching Pennies: 2 \n - Battle of the Sexes: 3 \n - Biased Rock, Paper, Scissors: 4 \n\n Game number is ")
    game = locals()["game{}".format(game_number)]

    env = rl_environment.Environment(game)
    num_actions = env.action_spec()["num_actions"]
    num_players = 2

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
            state_rate_self_play = eval_two_agents(env, agents[0], agents[1], 1000, int(game_number))
            state_rate_random = eval_two_agents(env, agents[0], random_agents[1], 1000, int(game_number))
            logging.info("Starting episode %s, state_rate_self_play %s, state_rate_random %s", cur_episode, state_rate_self_play, state_rate_random)
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
