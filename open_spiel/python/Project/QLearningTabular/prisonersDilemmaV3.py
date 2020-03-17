
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



def execute_scenarios(env, nb):
    sumScenario1 = np.zeros(4)
    for _ in range(nb):
        rewardCounter(sumScenario1 , scenario1(env))
    logging.info("The results of scenario 1 are: %s", sumScenario1)

    sumScenario2 = np.zeros(4)
    for _ in range(nb):
        rewardCounter(sumScenario2, scenario2(env))
    logging.info("The results of scenario 2 are: %s", sumScenario2)

    sumScenario3 = np.zeros(4)
    for _ in range(nb):
        rewardCounter(sumScenario3, scenario3(env))
    logging.info("The results of scenario 2 are: %s", sumScenario3)


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

"""
Scenario 2: 2 agents are trained with tabular_Q_learning against each other and evaluated against each other.
"""
def scenario2(env):
    agents = [
        tabular_qlearner.QLearner(player_id=idx, num_actions=env.num_actions_per_step, step_size=0.5, epsilon=0.2, discount_factor=1.0 )
        for idx in range(env.num_players)
    ]
    train_agents(env, agents, 1000)
    return evaluate_agents(env, agents)

"""
Scenario 3: 2 random against play against each other.
"""

def scenario3(env):
    random_agents = [
        random_agent.RandomAgent(player_id=idx, num_actions=env.num_actions_per_step, )
        for idx in range(env.num_players)
    ]

    return evaluate_agents(env, random_agents)



def rewardCounter(totalSum, reward):
    payOffMatrix = [[3,3],[0,5]], [[5,0],[1,1]]
    if reward == payOffMatrix[0][0]:
        totalSum[0] +=1
    elif reward == payOffMatrix[0][1]:
        totalSum[1] +=1
    elif reward == payOffMatrix[1][0]:
        totalSum[2] +=1
    elif reward == payOffMatrix[1][1]:
        totalSum[3] +=1



def main(_):
    game = create_game()
    env = create_environment(game)
    print(env.is_turn_based)
    execute_scenarios(env, 10)


if __name__ == "__main__":
  app.run(main)