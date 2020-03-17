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

flags.DEFINE_integer("num_episodes", int(5e4), "Number of train episodes.")
flags.DEFINE_boolean(
    "iteractive_play", True,
    "Whether to run an interactive play with the agent after training.")


def pretty_board(time_step):
  return


def command_line_action(time_step):
  return


def eval_against_random_bots(env, our_agent, other_agent, num_episodes):
    """Evaluates our agant against another against that both are tr for `num_episodes`."""
    wins = np.zeros(2)

    for _ in range(num_episodes):
        time_step = env.reset()
        while not time_step.last():
            agent_output = [our_agent.step(time_step, is_evaluation=True), other_agent.step(time_step, is_evaluation= True)]
            time_step = env.step([agent_output[0].action, agent_output[1].action])
        if time_step.rewards[0] == 1:
            wins[0] += 1
        elif (time_step.rewards[1] == 1) :
            wins[1] +=1


    return wins


def main(_):
    game = pyspiel.create_matrix_game("matching_pennies", "Matching Pennies",
                               ["Heads", "Tails"], ["Heads", "Tails"],
                               [[-1, 1], [1, -1]], [[1, -1], [-1, 1]])
    num_players = 2

    env = rl_environment.Environment(game)
    num_actions = env.action_spec()["num_actions"]

    our_agent = tabular_qlearner.QLearner(player_id=0, num_actions=num_actions)


    # random agents for evaluation
    other_agent = tabular_qlearner.QLearner(player_id=1, num_actions=num_actions)


    # 1. Train the agents
    training_episodes = FLAGS.num_episodes
    for cur_episode in range(training_episodes):
        if cur_episode % int(1e4) == 0:
            win_rates = eval_against_random_bots(env, our_agent, other_agent, 1000)
            logging.info("Starting episode %s, win_rates %s", cur_episode, win_rates)
        time_step = env.reset()
        while not time_step.last():
            actionAgent1 = our_agent.step(time_step, is_evaluation=False)
            actionAgent2 = other_agent.step(time_step, is_evaluation=False)
            agent_output = [actionAgent1, actionAgent2]
            time_step = env.step([agent_output[0].action, agent_output[1].action])


    if not FLAGS.iteractive_play:
        return


  # 2. Play from the command line against the trained agent.


if __name__ == "__main__":
  app.run(main)
