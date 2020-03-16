
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

def train_agents(env, agents, nbep):
    for ep in range(nbep):
        time_step = env.reset()
        while not time_step.last():
            actions = [agents[0].step(time_step).action, agents[1].step(time_step).action]
            time_step = env.step(actions)
        agents[0].step(time_step)
        agents[1].step(time_step)



def evaluate_agents(env, agents):
    time_step = env.reset()
    while not time_step.last():
        actions = [agents[0].step(time_step, is_evaluation = True).action, agents[1].step(time_step, is_evaluation = True).action]
        time_step = env.step(actions)

    agents[0].step(time_step, is_evaluation = True)
    agents[1].step(time_step, is_evaluation = True)
    return time_step.rewards

def create_game():
    return pyspiel.create_matrix_game("prisoners_dilemma", "Prisoners Dilemma",
                               ["coordinate", "betray"], ["coordinate", "betray"],
                               [[3,0],[5,1]], [[3,5],[0,1]])
def create_environment(game):
    return rl_environment.Environment(game)



def execute_scenarios(env):
    scenario1(env)

def scenario1(env):
    agents = [
        tabular_qlearner.QLearner(player_id=idx, num_actions=env.num_actions_per_step,)
        for idx in range(env.num_players)
    ]

    random_agents = [
        random_agent.RandomAgent(player_id=idx, num_actions=env.num_actions_per_step, )
        for idx in range(env.num_players)
    ]
    train_agents(env, [agents[0], random_agents[0]], 1000)
    env.reset()
    train_agents(env, [agents[1], random_agents[1]], 1000)

    print(evaluate_agents(env, agents))

def main(_):
    game = create_game()
    env = create_environment(game)
    for i in range(100):
        execute_scenarios(env)


if __name__ == "__main__":
  app.run(main)