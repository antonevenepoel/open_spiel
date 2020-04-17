
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
from open_spiel.python.algorithms import exploitability
from open_spiel.python.policy import PolicyFromCallable

FLAGS = flags.FLAGS

flags.DEFINE_integer("num_episodes", int(5e4), "Number of train episodes.")



def train_agents(env, agents):
    time_step = env.reset()
    while not time_step.last():
        player_id = time_step.observations["current_player"]
        agent_output = agents[player_id].step(time_step)
        time_step = env.step([agent_output.action])

    # Episode is over, step all agents with final info state.
    for agent in agents:
        agent.step(time_step)

def eval_against_other_agents(env, trained_agents, random_agents, num_episodes):
    """Evaluates `trained_agents` against `random_agents` for `num_episodes`."""
    wins = np.zeros(2)
    for player_pos in range(2):
        if player_pos == 0:
            cur_agents = [trained_agents[0], random_agents[1]]
        else:
            cur_agents = [random_agents[0], trained_agents[1]]
        for _ in range(num_episodes):
            time_step = env.reset()

            while not time_step.last():
                print(env._state.information_state_string())
                player_id = time_step.observations["current_player"]
                agent_output = cur_agents[player_id].step(time_step, is_evaluation=True)
                time_step = env.step([agent_output.action])

            if time_step.rewards[player_pos] >= 0:
                wins[player_pos] += 1
    return wins / num_episodes

def eval_against_themselves(env, agents, num_episodes):
    wins = np.zeros(2)
    for _ in range(num_episodes):
        time_step = env.reset()
        while not time_step.last():
            print(env._state.information_state_string())
            player_id = time_step.observations["current_player"]
            agent_output = agents[player_id].step(time_step, is_evaluation=True)
            time_step = env.step([agent_output.action])

        if time_step.rewards[0] >= 0:
            wins[0] += 1
        else:
            wins[1]+=1
    return wins / num_episodes

def main(_):
    game = pyspiel.load_game("kuhn_poker")

    num_players = 2

    env = rl_environment.Environment(game)
    num_actions = env.action_spec()["num_actions"]

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
            #win_rates = eval_against_other_agents(env, agents, random_agents, 1000)
            win_rates = eval_against_themselves(env, agents, 1000)
            PolicyFromCallable(game, agents[0]).action_probabilities()
            exploit = exploitability.exploitability(game, agents[0])
            logging.info("Starting episode %s, win_rates %s", cur_episode, win_rates)
        train_agents(env,agents)

if __name__ == "__main__":
    app.run(main)
