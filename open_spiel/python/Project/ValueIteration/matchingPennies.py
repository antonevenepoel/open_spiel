import pyspiel

from open_spiel.python.algorithms import value_iteration
from absl import app
import six

def playMatchingPennies():
    game = pyspiel.create_matrix_game("matching_pennies", "Matching Pennies",
                                      ["Heads", "Tails"], ["Heads", "Tails"],
                                      [[-1, 1], [1, -1]], [[1, -1], [-1, 1]])

    values = value_iteration.value_iteration(game, -1, 0.01)
    return values

def main(_):
    values = playMatchingPennies()
    print(values)
    for state, value in six.iteritems(values):
        print("")
        print(str(state))
        print("Value = {}".format(value))


if __name__ == "__main__":
  app.run(main)