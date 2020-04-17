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

"""NFSP agents trained on Kuhn Poker."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

# Disable all tensorflow warnings
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow

tensorflow.get_logger().setLevel('ERROR')
import tensorflow.compat.v1 as tf
from absl import logging

from open_spiel.python import policy
from open_spiel.python import rl_environment
from open_spiel.python.algorithms import exploitability
from open_spiel.python.algorithms import nfsp
import matplotlib.pyplot as plt
from open_spiel.python.project.part_2 import paths


class NFSPPolicies(policy.Policy):
    """Joint policy to be evaluated."""

    def __init__(self, env, nfsp_policies, mode):
        game = env.game
        player_ids = [0, 1]
        super(NFSPPolicies, self).__init__(game, player_ids)
        self._policies = nfsp_policies
        self._mode = mode
        self._obs = {"info_state": [None, None], "legal_actions": [None, None]}

    def action_probabilities(self, state, player_id=None):
        cur_player = state.current_player()
        legal_actions = state.legal_actions(cur_player)

        self._obs["current_player"] = cur_player
        self._obs["info_state"][cur_player] = (
            state.information_state_tensor(cur_player))
        self._obs["legal_actions"][cur_player] = legal_actions

        info_state = rl_environment.TimeStep(
            observations=self._obs, rewards=None, discounts=None, step_type=None)

        with self._policies[cur_player].temp_mode_as(self._mode):
            p = self._policies[cur_player].step(info_state, is_evaluation=True).probs
        prob_dict = {action: p[action] for action in legal_actions}
        return prob_dict


def train_nfsp(
        game="kuhn_poker",
        num_players=2,
        hidden_layers_sizes=(64,),
        replay_buffer_capacity=int(2e5),
        num_train_episodes=int(1e4),
        epsilon_start=0.06,
        epsilon_end=0.001,
        reservoir_buffer_capacity=int(2e6),
        anticipatory_param=0.1,
        eval_every=int(1e4),
        batch_size=128,
        rl_learning_rate=0.01,
        sl_learning_rate=0.01,
        min_buffer_size_to_learn=1000,
        learn_every=64,
        optimizer_str="sgd"
) -> dict:

    data = {
        "game": game,
        "players": num_players,
        "hidden_layers_sizes": hidden_layers_sizes,
        "replay_buffer_capacity": replay_buffer_capacity,
        "num_train_episodes": num_train_episodes,
        "epsilon_start" : epsilon_start,
        "epsilon_end": epsilon_end,
        "reservoir_buffer_capacity": reservoir_buffer_capacity,
        "anticipatory_param": anticipatory_param,
        "eval_every": eval_every,
        "batch_size": batch_size,
        "rl_learning_rate": rl_learning_rate,
        "sl_learning_rate": sl_learning_rate,
        "min_buffer_size_to_learn": min_buffer_size_to_learn,
        "learn_every": learn_every,
        "optimizer_str": optimizer_str,
        "episodes": [],
        "losses": [],
        "exploitability": []
    }
    env_configs = {"players": num_players}
    env = rl_environment.Environment(game, **env_configs)
    info_state_size = env.observation_spec()["info_state"][0]
    num_actions = env.action_spec()["num_actions"]

    hidden_layers_sizes = [int(l) for l in hidden_layers_sizes]  # FLAGS.hidden_layers_sizes]
    kwargs = {
        "replay_buffer_capacity": replay_buffer_capacity,  # FLAGS.replay_buffer_capacity,
        "epsilon_decay_duration": num_train_episodes,  # FLAGS.num_train_episodes,
        "epsilon_start": epsilon_start,  # FLAGS.epsilon_start,
        "epsilon_end": epsilon_end,  # FLAGS.epsilon_end,
        "batch_size": batch_size,
        "rl_learning_rate": rl_learning_rate,
        "sl_learning_rate": sl_learning_rate,
        "min_buffer_size_to_learn": min_buffer_size_to_learn,
        "learn_every": learn_every,
        "optimizer_str": optimizer_str
    }

    with tf.Session() as sess:
        # pylint: disable=g-complex-comprehension
        agents = [
            nfsp.NFSP(
                sess,
                idx,
                info_state_size,
                num_actions,
                hidden_layers_sizes,
                reservoir_buffer_capacity,
                anticipatory_param,
                **kwargs
            ) for idx in range(num_players)
        ]
        expl_policies_avg = NFSPPolicies(env, agents, nfsp.MODE.average_policy)
        sess.run(tf.global_variables_initializer())
        for ep in range(num_train_episodes):  # FLAGS.num_train_episodes):
            if (ep+1) % eval_every == 0 or ep == 0:  # FLAGS.eval_every == 0:
                losses = [agent.loss for agent in agents]
                logging.info("Losses: %s", losses)
                expl = exploitability.exploitability(env.game, expl_policies_avg)
                print("Exploitability AVG", ep+1, expl)
                print("_____________________________________________")
                # logging.info("[%s] Exploitability AVG %s", ep + 1, expl)
                # logging.info("_____________________________________________")

                # store info
                data["exploitability"].append(expl)
                data["losses"].append(losses)
                data["episodes"].append(ep+1)

            time_step = env.reset()
            while not time_step.last():
                player_id = time_step.observations["current_player"]
                agent_output = agents[player_id].step(time_step)
                action_list = [agent_output.action]
                time_step = env.step(action_list)

            # Episode is over, step all agents with final info state.
            for agent in agents:
                agent.step(time_step)

    sess.close()
    return data
