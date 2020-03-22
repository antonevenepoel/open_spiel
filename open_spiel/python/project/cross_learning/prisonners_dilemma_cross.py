
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
from open_spiel.python.project.cross_learning import cross_learner



def train_agents(env, agents, nbep):
    prob_array = []
    for ep in range(nbep):
        time_step = env.reset()
        while not time_step.last():
            output0 = agents[0].step(time_step)
            output1 = agents[1].step(time_step)
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
    return pyspiel.create_matrix_game([[3/5,0],[1,1/5]], [[3/5,1],[0,1/5]])
def create_environment(game):
    return rl_environment.Environment(game)




def execute_scenarios_probs(game, nb):
    env = create_environment(game)

    agents = [
        tabular_qlearner.QLearner(player_id=0, num_actions=env.num_actions_per_step, step_size=0.1, epsilon=0.2, discount_factor=1.0 ),
        tabular_qlearner.QLearner(player_id=1, num_actions=env.num_actions_per_step, step_size=0.1, epsilon=0.2, discount_factor=1.0)
    ]

    cross_agents = [
        cross_learner.CrossLearner(player_id=0, num_actions=env.num_actions_per_step),
        cross_learner.CrossLearner(player_id=0, num_actions=env.num_actions_per_step)
    ]

    random_agents = [
        random_agent.RandomAgent(player_id=1, num_actions=env.num_actions_per_step, ),
        random_agent.RandomAgent(player_id=0, num_actions=env.num_actions_per_step, )
    ]

    list = train_agents(env, [cross_agents[0], cross_agents[1]], 1000)


    return list


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

    list = execute_scenarios_probs(game, 1000)
    print(list)



if __name__ == "__main__":
  app.run(main)