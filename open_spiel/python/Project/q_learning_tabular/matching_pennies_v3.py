
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
from absl import app
import numpy as np

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
    return pyspiel.create_matrix_game("matching_pennies", "Matching Pennies",
                                      ["Heads", "Tails"], ["Heads", "Tails"],
                                      [[-1, 1], [1, -1]], [[1, -1], [-1, 1]])
def create_environment(game):
    return rl_environment.Environment(game)


def execute_scenarios(env, nb):
    sumScenario1 = np.zeros(2)
    for _ in range(nb):
        rewardCounter(sumScenario1 , scenario1(env))
    logging.info("The results of scenario 1 are: %s", sumScenario1)


"""
Scenario 1: 2 agents are trained with tabular_Q_learning against a random agent and evaluated against each other.
"""
def scenario1(env):
    agents = [
        tabular_qlearner.QLearner(player_id=idx, num_actions=env.num_actions_per_step, )
        for idx in range(env.num_players)
    ]
    random_agents = [
        random_agent.RandomAgent(player_id=idx, num_actions=env.num_actions_per_step, )
        for idx in range(env.num_players)
    ]
    train_agents(env, [agents[0], random_agents[0]], 1000)
    train_agents(env, [agents[1], random_agents[1]], 1000)

    return evaluate_agents(env, agents)


def rewardCounter(totalSum, reward):
    if reward == [1.0, -1.0]:
        totalSum[0] +=1
    elif reward == [-1.0, 1.0]:
        totalSum[1] +=1


def main(_):
    game = create_game()
    env = create_environment(game)
    print(env.is_turn_based)
    execute_scenarios(env, 100)


if __name__ == "__main__":
  app.run(main)