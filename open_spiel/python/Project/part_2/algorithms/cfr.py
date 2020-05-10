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

"""Example use of the CFR algorithm on Kuhn Poker."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from absl import app
from absl import flags

from open_spiel.python.algorithms import cfr
from open_spiel.python.algorithms import exploitability
import pyspiel
import matplotlib.pyplot as plt

from open_spiel.python.algorithms.cfr import _CFRSolver

# FLAGS = flags.FLAGS

# flags.DEFINE_integer("iterations", 100, "Number of iterations")
# flags.DEFINE_string("game", "kuhn_poker", "Name of the game")
# flags.DEFINE_integer("players", 2, "Number of players")
# flags.DEFINE_integer("print_freq", 10, "How often to print the exploitability")
#from open_spiel.python.project.ML_project20.tournament import policy_to_csv
from open_spiel.python.project.part_2 import path_file


def train_cfr(
        game_name: str,
        regret_matching_plus: bool,
        alternating_updates: bool,
        linear_averaging: bool,
        average_policy: bool,
        iterations: int,
        print_freq: int,
        players: int = 2,
        modeldir = "./models"
) -> dict:
    data = {
        "players": players,
        "game": game_name,
        "regret_matching_plus": regret_matching_plus,
        "alternating_updates": alternating_updates,
        "linear_averaging": linear_averaging,
        "average_policy" : average_policy,
        "print_freq": print_freq,
        "exploitability": [],
        "iterations": []
    }

    game = pyspiel.load_game(game_name,
                             {"players": pyspiel.GameParameter(players)})
    cfr_solver = _CFRSolver(
        game=game,
        linear_averaging=linear_averaging,
        regret_matching_plus=regret_matching_plus,
        alternating_updates=alternating_updates
    )

    expl_policies = None
    for i in range(iterations):  # FLAGS.iterations):
        cfr_solver.evaluate_and_update_policy()
        if (i + 1) % print_freq == 0 or i == 0:  # FLAGS.print_freq == 0:
            expl_policies = cfr_solver.average_policy() if average_policy else cfr_solver.current_policy()
            conv = exploitability.exploitability(game, expl_policies)
            print("Iteration {} exploitability {}".format(i + 1, conv))

            # store info
            data["exploitability"].append(conv)
            data["iterations"].append(i + 1)

    # save policies to csv
    policy_to_csv(game, expl_policies, f"{modeldir}/test_p{1}_{game_name}.csv")

    return data
