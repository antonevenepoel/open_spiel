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

flags.DEFINE_integer("num_episodes", int(1e1), "Number of train episodes.")
flags.DEFINE_boolean(
    "iteractive_play", True,
    "Whether to run an interactive play with the agent after training.")


def pretty_board(time_step):
  """Returns the board in `time_step` in a human readable format."""
  info_state = time_step.observations["info_state"][0]
  x_locations = np.nonzero(info_state[9:18])[0]
  o_locations = np.nonzero(info_state[18:])[0]
  board = np.full(3 * 3, ".")
  board[x_locations] = "X"
  board[o_locations] = "0"
  board = np.reshape(board, (3, 3))
  return board


def command_line_action(time_step):
  """Gets a valid action from the user on the command line."""
  current_player = time_step.observations["current_player"]
  legal_actions = time_step.observations["legal_actions"][current_player]
  action = -1
  while action not in legal_actions:
    print("Choose an action from {}:".format(legal_actions))
    sys.stdout.flush()
    action_str = input()
    try:
      action = int(action_str)
    except ValueError:
      continue
  return action

def eval_against_random_bots(env, trained_agents, random_agents, num_episodes):
    """Evaluates `trained_agents` against `random_agents` for `num_episodes`."""
    wins = np.zeros(4)
    cur_agents = trained_agents
    for _ in range(num_episodes):
        time_step = env.reset()
        while not time_step.last():
            action0 = cur_agents[0].step(time_step, is_evaluation= True).action
            action1 = cur_agents[1].step(time_step, is_evaluation= True).action
            time_step = env.step([action0, action1])

        # print(time_step.rewards)

        if time_step.rewards == [-1, -1]:
            wins[0] += 1
        elif (time_step.rewards == [0, -10]):
            wins[1] += 1
        elif (time_step.rewards == [-10, 0]):
            wins[2] += 1
        elif (time_step.rewards == [-5, -5]):
            wins[3] += 1

        for agent in cur_agents:
            agent.step(time_step)

    return wins / num_episodes

def  eval_against_each_other(env, agent0, agent1, num_episodes):
    """Evaluates `trained_agents` against `random_agents` for `num_episodes`."""
    wins = np.zeros(4)
    for _ in range(num_episodes):
        time_step = env.reset()
        while not time_step.last():
            action0 = agent0.step(time_step, is_evaluation=True).action
            action1 = agent1.step(time_step, is_evaluation=True).action
            time_step = env.step([action0, action1])
        agent0.step(time_step)
        agent1.step(time_step)
        if time_step.rewards == [-1, -1]:
            wins[0] += 1
        elif (time_step.rewards == [0, -10]):
            wins[1] += 1
        elif (time_step.rewards == [-10, 0]):
            wins[2] += 1
        elif (time_step.rewards == [-5, -5]):
            wins[3] += 1
    return wins


def main(_):
    game = pyspiel.create_matrix_game("prisoners_dilemma", "Prisoners Dilemma",
                               ["Confess", "Silent"], ["Confess", "Silent"],
                               [[3, 0], [5, 1]],
                                [[3, 5], [0, 1]])

    num_players = 2

    env = rl_environment.Environment(game)
    num_actions = env.action_spec()["num_actions"]
    agents = [
        tabular_qlearner.QLearner(player_id=idx, num_actions=num_actions, step_size=0.5, epsilon=0.2, discount_factor=1.0)
        for idx in range(num_players)
    ]
    # random agents for evaluation
    random_agents = [
        random_agent.RandomAgent(player_id=idx, num_actions=num_actions)
        for idx in range(num_players)
    ]


    # # 1. Train the agents
    # training_episodes = FLAGS.num_episodes
    # for cur_episode in range(training_episodes):
    #     if cur_episode % int(1e4) == 0:
    #         win_rates = eval_against_random_bots(env, agents, random_agents, 1000)
    #         logging.info("Starting episode %s, win_rates %s", cur_episode, win_rates)
    #     for player_pos in range(2):
    #         time_step = env.reset()
    #         while not time_step.last():
    #
    #             action0 = agents[player_pos].step(time_step).action
    #
    #             action1 = random_agents[0].step(time_step).action
    #
    #             time_step = env.step([action0, action1])
    #
    #         agents[player_pos].step(time_step)
    #         random_agents[player_pos].step(time_step)

    for ep in range(10):
        for ep2 in range(1000):
            # training
            for pos in range(2):
                time_step = env.reset()
                while not time_step.last():
                    action0 = agents[pos].step(time_step).action
                    action1 = random_agents[pos].step(time_step).action
                    actionList= [action0, action1]
                    time_step = env.step(actionList)
                agents[pos].step(time_step)
                random_agents[pos].step(time_step)

        win_rates = eval_against_each_other(env, agents[0],agents[1], 1000)
        logging.info("Starting episode %s, win_rates %s", ep, win_rates)

if __name__ == "__main__":
  app.run(main)
