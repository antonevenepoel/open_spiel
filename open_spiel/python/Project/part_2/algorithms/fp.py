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

"""Python XFP example."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

import numpy as np
from absl import app
from absl import flags

from open_spiel.python import policy
from open_spiel.python.algorithms import exploitability
from open_spiel.python.algorithms import fictitious_play
import pyspiel
from open_spiel.python.project.part_2 import path_file
import matplotlib.pyplot as plt


def train_fp(
        game="kuhn_poker",
        players=2,
        print_freq=int(1e3),
        iterations=int(1e4)
):
    data = {
        "game": game,
        "players": players,
        "print_freq": print_freq,
        "iterations": [],
        "exploitability": []
    }
    game = pyspiel.load_game(game,
                             {"players": pyspiel.GameParameter(players)})
    xfp_solver = fictitious_play.XFPSolver(game)
    for i in range(iterations):
        xfp_solver.iteration()
        conv = exploitability.exploitability(
            game,
            policy.PolicyFromCallable(game, xfp_solver.average_policy_callable()))
        if (i+1) % print_freq == 0 or i == 0:
            print("Iteration: {} Conv: {}".format(i+1, conv))
            sys.stdout.flush()

            # add info
            data["iterations"].append(i+1)
            data["exploitability"].append(conv)

    return data
