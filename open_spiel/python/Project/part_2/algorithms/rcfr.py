from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
import tensorflow.compat.v1 as tf

from open_spiel.python.algorithms import rcfr
import pyspiel

tf.enable_eager_execution()



def train_rcfr(
        game_name="kuhn_poker",
        eval_every=int(1e2),
        num_iterations=int(1e3),
        num_players = 2,
        bootstrap=False,
        truncate_negative = False,
        buffer_size = -1,
        num_hidden_layers=1,
        num_hidden_units=13,
        num_hidden_factors = 8,
        use_skip_connections = True,
        num_epochs = 200,
        batch_size=100,
        step_size=0.01
):
    data = {
        "game": game_name,
        "eval_every": eval_every,
        "num_iterations": num_iterations,
        "num_players": num_players,
        "bootstrap" :bootstrap,
        "truncate_negative" : truncate_negative,
        "buffer_size": buffer_size ,
        "num_hidden_layers": num_hidden_layers,
        "num_hidden_units": num_hidden_units,
        "num_hidden_factors": num_hidden_factors,
        "use_skip_connections": use_skip_connections,
        "num_epochs":  num_epochs,
        "batch_size":  batch_size,
        "step_size": step_size,
        "iterations": [],
        "exploitability": [],

    }


    game = pyspiel.load_game(game_name,
                             {"players": pyspiel.GameParameter(num_players)})

    models = []
    for _ in range(game.num_players()):
        models.append(
            rcfr.DeepRcfrModel(
                game,
                num_hidden_layers=num_hidden_layers,
                num_hidden_units=num_hidden_units,
                num_hidden_factors=num_hidden_factors,
                use_skip_connections=use_skip_connections))

    if buffer_size > 0:
        solver = rcfr.ReservoirRcfrSolver(
            game,
            models,
            buffer_size,
            truncate_negative=truncate_negative)
    else:
        solver = rcfr.RcfrSolver(
            game,
            models,
            truncate_negative=truncate_negative,
            bootstrap=bootstrap)

    def _train_fn(model, data):
        """Train `model` on `data`."""
        data = data.shuffle(batch_size * 10)
        data = data.batch(batch_size)
        data = data.repeat(num_epochs)

        optimizer = tf.keras.optimizers.Adam(lr=step_size, amsgrad=True)

        @tf.function
        def _train():
            for x, y in data:
                optimizer.minimize(
                    lambda: tf.losses.huber_loss(y, model(x), delta=0.01),  # pylint: disable=cell-var-from-loop
                    model.trainable_variables)

        _train()

    # End of _train_fn

    for i in range(num_iterations):
        solver.evaluate_and_update_policy(_train_fn)
        if i % eval_every == 0:
            conv = pyspiel.exploitability(game, solver.average_policy())
            print("Iteration {} exploitability {}".format(i, conv))

        data["exploitability"].append(conv)
        data["iterations"].append(i + 1)
    return data