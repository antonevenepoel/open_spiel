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
    prob_array = []
    for ep in range(nbep):
        time_step = env.reset()
        while not time_step.last():
            output0 = agents[0].step(time_step)
            output1 = agents[1].step(time_step)
            prob_array = []
            prob_array.append(output0.probs)
            prob_array.append(output1.probs)
            actions = [output0.action, output1.action]
            time_step = env.step(actions)
        # Episode is over, step all agents with final info state.
        agents[0].step(time_step)
        agents[1].step(time_step)
#        print(time_step.rewards)
    return prob_array


def evaluate_agents(env, agents):
    time_step = env.reset()
    while not time_step.last():
        actions = [agents[0].step(time_step, is_evaluation = True).action, agents[1].step(time_step, is_evaluation = True).action]
        time_step = env.step(actions)
    return time_step.rewards


def create_game():
    return pyspiel.create_matrix_game([[3,0],[5,1]], [[3,5],[0,1]])
def create_environment(game):
    return rl_environment.Environment(game)


def execute_scenarios(env, nb, nbsc):

    if nbsc == 1 or nbsc == 0:
        sumScenario1 = np.zeros(4)
        for _ in range(nb):
            rewardCounter(sumScenario1 , scenario1(env))
        logging.info("The results of scenario 1 are: %s", sumScenario1)
    if nbsc == 2 or nbsc == 0:
        sumScenario2 = np.zeros(4)
        for _ in range(nb):
            rewardCounter(sumScenario2, scenario2(env))
        logging.info("The results of scenario 2 are: %s", sumScenario2)

    if nbsc == 2 or nbsc == 0:
        sumScenario3 = np.zeros(4)
        for _ in range(nb):
            rewardCounter(sumScenario3, scenario3(env))
        logging.info("The results of scenario 2 are: %s", sumScenario3)



def execute_scenarios_probs(game, nb):
    env = create_environment(game)


    agents = [
        tabular_qlearner.QLearner(player_id=0, num_actions=env.num_actions_per_step, step_size=0.1, epsilon=0.2, discount_factor=1.0 ),
        tabular_qlearner.QLearner(player_id=1, num_actions=env.num_actions_per_step, step_size=0.1, epsilon=0.2, discount_factor=1.0)

    ]
    random_agents = [
        random_agent.RandomAgent(player_id=1, num_actions=env.num_actions_per_step, ),
        random_agent.RandomAgent(player_id=0, num_actions=env.num_actions_per_step, )

    ]
    dictionary = dict()


    dictionary[0] =  train_agents(env, [agents[0], random_agents[0]], 1000)
    dictionary[1] = train_agents(env, [random_agents[1], agents[1]], 1000)
    return dictionary
























"""
Scenario 1: 2 agents are trained with tabular_Q_learning against a random agent and evaluated against each other.
"""
def scenario1(env):
    agents = [
        tabular_qlearner.QLearner(player_id=0, num_actions=env.num_actions_per_step, step_size=0.1, epsilon=0.2, discount_factor=1.0 ),
        tabular_qlearner.QLearner(player_id=0, num_actions=env.num_actions_per_step, step_size=0.1, epsilon=0.2, discount_factor=1.0)

    ]
    random_agents = [
        random_agent.RandomAgent(player_id=1, num_actions=env.num_actions_per_step, ),
        random_agent.RandomAgent(player_id=1, num_actions=env.num_actions_per_step, )

    ]
    train_agents(env, [agents[0], random_agents[0]], 1000)
    train_agents(env, [agents[0], random_agents[1]], 1000)

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
 #   execute_scenarios(env, 10, 1)

    dict = execute_scenarios_probs(game, 1000)
    print(dict[0][len(dict[0]) - 1 ])
    print(dict[1][len(dict[1]) - 1])



if __name__ == "__main__":
  app.run(main)