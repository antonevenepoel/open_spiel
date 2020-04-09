
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
import numpy as np

import pyspiel

from open_spiel.python import rl_environment
from open_spiel.python.project.part_1.cross_learning import cross_learner
from open_spiel.python.algorithms import random_agent
from open_spiel.python.algorithms import tabular_qlearner

ACCURACY = 10000
ITERATIONS = 1000


def train_agents(env, agents, nbep, i):
    prob_array = []
    for ep in range(nbep):
        time_step = env.reset()
        while not time_step.last():
            output0 = agents[0].step(time_step)
            output1 = agents[1].step(time_step)
            if i ==0:
                prob_array += [output0.probs[0]]
            else:
                prob_array += [output1.probs[0]]

            actions = [output0.action, output1.action]
            time_step = env.step(actions)
        # Episode is over, step all agents with final info state.
        agents[0].step(time_step)
        agents[1].step(time_step)
#        print(time_step.rewards)
    return prob_array

def train_agents_simultaneous(env, agents, nbep):
    prob_array = []
    prob_array_agent_1=[]
    prob_array_agent_2 =[]
    for ep in range(nbep):
        time_step = env.reset()
        while not time_step.last():
            output0 = agents[0].step(time_step)
            output1 = agents[1].step(time_step)
            prob_array_agent_1 += [output0.probs[0]]
            prob_array_agent_2 += [output1.probs[0]]

            actions = [output0.action, output1.action]
            time_step = env.step(actions)
        # Episode is over, step all agents with final info state.
        agents[0].step(time_step)
        agents[1].step(time_step)
    prob_array.append(prob_array_agent_1)
    prob_array.append(prob_array_agent_2)
    return prob_array


def train_agents_simultaneous_single(env, agents, nbep):
    prob_array = []

    for ep in range(nbep):
        time_step = env.reset()
        while not time_step.last():
            output0 = agents[0].step(time_step)
            output1 = agents[1].step(time_step)
            prob_array.append([output0.probs[0], output0.probs[1], output0.probs[2]])

            actions = [output0.action, output1.action]
            time_step = env.step(actions)
        # Episode is over, step all agents with final info state.
        agents[0].step(time_step)
        agents[1].step(time_step)
    return prob_array


def evaluate_agents(env, agents):
    time_step = env.reset()
    while not time_step.last():
        actions = [agents[0].step(time_step, is_evaluation = True).action, agents[1].step(time_step, is_evaluation = True).action]
        time_step = env.step(actions)
    return time_step.rewards


def create_game(name):
    if name == "PD":
        return pyspiel.create_matrix_game([[3/5,0],[1,1/5]], [[3/5,1],[0,1/5]])
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
            [[0.5, 0.25, 1], [0.75, 0.5, 0.45], [0, 0.55, 0.5]],
            [[0.5, 0.75, 0], [0.25, 0.5, 0.55], [1, 0.45, 0.5]])

def create_environment(game):
    return rl_environment.Environment(game)




def execute_scenarios_probs(env, nb, start, sc):
    agents = [
        tabular_qlearner.QLearner(player_id=0, num_actions=env.num_actions_per_step, step_size=0.5, epsilon=0.05, discount_factor=1.0 ),
        tabular_qlearner.QLearner(player_id=1, num_actions=env.num_actions_per_step, step_size=0.5, epsilon=0.05, discount_factor=1.0)

    ]
    random_agents = [
        random_agent.RandomAgent(player_id=1, num_actions=env.num_actions_per_step, ),
        random_agent.RandomAgent(player_id=0, num_actions=env.num_actions_per_step, )

    ]

    agents[0]._q_values['[0.0]'][0] = start[0]
    agents[1]._q_values['[0.0]'][0] = start[1]

    if sc == 0:
        list = []
        list.append(train_agents(env, [agents[0], random_agents[0]], nb, 0))
        list.append(train_agents(env, [random_agents[1], agents[1]], nb, 1))
        return list
    elif sc == 1:
        return train_agents_simultaneous(env, agents, nb)
    elif sc ==2:
        cross_agents = [
            cross_learner.CrossLearner(player_id=0, num_actions=env.num_actions_per_step, step_size=0.001),
            cross_learner.CrossLearner(player_id=1, num_actions=env.num_actions_per_step, step_size=0.001)
        ]
        cross_agents[0]._probs = [start[0], 1 - start[0]]
        cross_agents[1]._probs = [start[1], 1 - start[1]]
        return train_agents_simultaneous(env, cross_agents, nb)
    else:
        cross_agents = [
            cross_learner.CrossLearner(player_id=0, num_actions=3, step_size=0.001),
            cross_learner.CrossLearner(player_id=1, num_actions=3, step_size=0.001)
        ]
        cross_agents[0]._probs = [start[0][0], start[0][1], 1-start[0][0]-start[0][1]]
        cross_agents[1]._probs = [start[1][0], start[1][1], 1-start[1][0]-start[1][1]]
        return train_agents_simultaneous_single(env, cross_agents, nb)



def average_prob(env, start, sc):
    nb = ITERATIONS
    list = []
    for i in range(ACCURACY):
        list.append(execute_scenarios_probs(env, nb, start, sc))

    prob1 = np.zeros(nb)
    prob2 = np.zeros(nb)
    for elem in list:

        prob1 = [sum(pair) for pair in zip(prob1, elem[0])]
        prob2 = [sum(pair) for pair in zip(prob2, elem[1])]

    return [(x/ACCURACY, y/ACCURACY) for (x, y) in zip(prob1, prob2)]


def prepare_plot(name, start=(0,0), sc=0):
    game = create_game(name)
    env = create_environment(game)
    if sc != 3:
        return average_prob(env, start, sc)
    else:
        return execute_scenarios_probs(env, ITERATIONS, start, sc)


def main(_):
    game = create_game("RPS")
    env = create_environment(game)
    print(env.num_actions_per_step)
    return execute_scenarios_probs(env, ITERATIONS, ((0.3 , 0.3), (0.3 , 0.3)), 3)



if __name__ == "__main__":
  app.run(main)