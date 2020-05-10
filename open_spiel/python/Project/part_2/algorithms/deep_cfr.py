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

"""Python Deep CFR example."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pyspiel
import six
from absl import app

from open_spiel.python.algorithms import deep_cfr
from open_spiel.python.algorithms import exploitability

import collections
import random
import numpy as np
import sonnet as snt
import os

# Disable all tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow

tensorflow.get_logger().setLevel('ERROR')
import tensorflow.compat.v1 as tf

from open_spiel.python import policy


def train_dcfr(
        game_name="kuhn_poker",
        eval_every=int(1e2),
        num_iterations=int(1e3),
        num_traversals=40,
        policy_network_layers=(32, 32),
        advantage_network_layers=(16, 16),
        learning_rate=1e-3,
        batch_size_advantage=None,
        batch_size_strategy=None,
        memory_capacity=1e7
):
    data = {
        "game": game_name,
        "eval_every": eval_every,
        "num_iterations": num_iterations,
        "num_traversals": num_traversals,
        "policy_network_layers": policy_network_layers,
        "advantage_network_layers": advantage_network_layers,
        "learning_rate": learning_rate,
        "batch_size_advantage": batch_size_advantage,
        "batch_size_strategy": batch_size_strategy,
        "memory_capacity": memory_capacity,
        "iterations": [],
        "exploitability": []
    }

    game = pyspiel.load_game(game_name)
    for iteration in range(num_iterations):
        if (iteration+1) % eval_every == 0 or iteration == 1:
            with tf.Session() as sess:
                deep_cfr_solver = deep_cfr.DeepCFRSolver(
                    sess,
                    game,
                    policy_network_layers,
                    advantage_network_layers,
                    iteration,
                    num_traversals,
                    learning_rate,
                    batch_size_advantage,
                    batch_size_strategy,
                    memory_capacity)
                sess.run(tf.global_variables_initializer())
                _, advantage_losses, policy_loss = deep_cfr_solver.solve()
                # for player, losses in six.iteritems(advantage_losses):
                    # print("Advantage for player %s: %s"
                    #      % (player, losses[:2] + ["..."] + losses[-2:]))
                    # print("Advantage Buffer Size for player %s: '%s'"
                    #      % (player, len(deep_cfr_solver.advantage_buffers[player])))
                # print("Strategy Buffer Size: '%s'"
                #      % len(deep_cfr_solver.strategy_buffer))
                # print("Final policy loss: '%s'" % policy_loss)
                conv = exploitability.nash_conv(
                    game,
                    policy.PolicyFromCallable(game, deep_cfr_solver.action_probabilities))
                print("Number of iterations %s - NashConv: %s" % (iteration+1, conv))

                # Store the data
                data["exploitability"].append(conv)
                data["iterations"].append(iteration+1)
    return data

