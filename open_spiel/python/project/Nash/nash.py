
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
def train_agents_simultaneous_single(env, agents, nbep):
    for ep in range(nbep):
        time_step = env.reset()
        while not time_step.last():
            output0 = agents[0].step(time_step)
            output1 = agents[1].step(time_step)
            actions = [output0.action, output1.action]
            time_step = env.step(actions)
        # Episode is over, step all agents with final info state.
        agents[0].step(time_step)
        agents[1].step(time_step)



def evaluate_agents(env, agents):
    time_step = env.reset()
    while not time_step.last():
        output0 = agents[0].step(time_step, is_evaluation = True)
        output1 = agents[1].step(time_step)
        action000 = output0.action
        actions = [output0.action, output1.action]
        time_step = env.step(actions)
    # Episode is over, step all agents with final info state.
    agents[0].step(time_step)
    agents[1].step(time_step)
    return action000


def create_game(name):
    if name == "PD":
        return pyspiel.create_matrix_game([[3,0],[5,1]], [[3,5],[0,1]])
    elif name == "BOS":
        return pyspiel.create_matrix_game("battle_of_sexes", "The Battle of The Sexes",
                                          ["LW", "WL"], ["LW", "WL"],
                                          [[1, 0], [0, 1/2]], [[1/2, 0], [0, 1]])
    elif name == "MP":
        return pyspiel.create_matrix_game("matching_pennies", "Matching Pennies",
                                          ["Heads", "Tails"], ["Heads", "Tails"],
                                          [[0, 1], [1, 0]], [[1, 0], [0, 1]])
    elif name == "RPS":
        return pyspiel.create_matrix_game(
            [[0.0, -0.25, 0.5], [0.25, 0.0, -0.05], [-0.5, 0.05, 0.0]],
            [[0.0, 0.25, -0.5], [-0.25, 0.0, 0.05], [0.5, -0.05, 0.0]])
def create_environment(game):
    return rl_environment.Environment(game)



def execute_scenarios_probs(env, nb, start):
    agents = [
        tabular_qlearner.QLearner(player_id=0, num_actions=3, step_size=0.5, epsilon=0.05, discount_factor=1.0 ),
        tabular_qlearner.QLearner(player_id=1, num_actions=3, step_size=0.5, epsilon=0.05, discount_factor=1.0)

    ]
    agents[0]._q_values['[0.0]'][0] = start[0]
    agents[1]._q_values['[0.0]'][0] = start[1]
    train_agents_simultaneous_single(env, agents, nb)
    return evaluate_agents(env, agents)


def rewardCounter(totalSum, reward, payOffMatrix):

    if reward == payOffMatrix[0][0]:
        totalSum[0] +=1
    elif reward == payOffMatrix[0][1]:
        totalSum[1] +=1
    elif reward == payOffMatrix[1][0]:
        totalSum[2] +=1
    elif reward == payOffMatrix[1][1]:
        totalSum[3] +=1


def rewardCounter2(totalSum, reward, payOffMatrix):

    if reward == payOffMatrix[0][0]:
        totalSum[0][0] +=1
    elif reward == payOffMatrix[0][1]:
        totalSum[0][1] +=1
    elif reward == payOffMatrix[0][2]:
        totalSum[0][2] +=1
    elif reward == payOffMatrix[1][0]:
        totalSum[1][0] +=1
    elif reward == payOffMatrix[1][1]:
        totalSum[1][1] +=1
    elif reward == payOffMatrix[1][2]:
        totalSum[1][2] +=1
    elif reward == payOffMatrix[2][0]:
        totalSum[2][0] +=1
    elif reward == payOffMatrix[2][1]:
        totalSum[2][1] +=1
    elif reward == payOffMatrix[2][2]:
        totalSum[2][2] += 1

def rewardCounter3(totalSum, reward, qsdf):
    if reward == 0:
        totalSum[0] +=1
    elif reward == 1:
        totalSum[1] +=1
    elif reward == 2:
        totalSum[2] += 1


def create_payoff(name):
    if name == "PD":
        return [[3,3],[0,5]], [[5,0],[1,1]]
    if name == "BOS":
        return [[1,1/2],[0,0]], [[0,0],[1/2,1]]
    if name == "RPS":
        return [[[0,0],[-0.25,25],[0.5,-0.5]], [[0.25,-0.25],[0,0],[-0.05,0.05]], [[-0.5,0.5],[0.05,-0.05],[0,0]]]
    if name == "MP":
        return [[0,1],[1,0]], [[1,0],[0,1]]


def main(_):
    name = "RPS"
    game = create_game(name)
    payoff = create_payoff(name)
    env = create_environment(game)
    totalsum = np.zeros(4)
    #sum = [[0,0,0],[0,0,0],[0,0,0]]
    for i in range(1000):
        print(i)
        rewardCounter3(totalsum, execute_scenarios_probs(env, 1000, (0, 0)), payoff)
    print(totalsum)


if __name__ == "__main__":
  app.run(main)