
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import numpy as np

from open_spiel.python import rl_agent

class CrossLearner(rl_agent.AbstractAgent):

    def __init__(self,
                 player_id,
                 num_actions,
                 step_size=0.5,
                 ):
        """Initialize the Q-Learning agent."""
        self._player_id = player_id
        self._num_actions = num_actions
        self._step_size = step_size
        self._probs = [1/num_actions for i in range(num_actions)]
        self.prev_action = None


    def step(self, time_step, is_evaluation=False):
        action, probs = None, None
        # Act step: don't act at terminal states.
        if not time_step.last():
            action = np.random.choice(range(self._num_actions), p=self._probs)
            probs = self._probs

        if self.prev_action != None:
            reward = time_step.rewards[self._player_id]
            self.update_prob(reward)
            self.prev_action = None
        self.prev_action = action
        return rl_agent.StepOutput(action=action, probs=probs)

    def update_prob(self, reward):
        action = self.prev_action
        self._probs[action] += self._step_size*(reward - self._probs[action]*reward)
        for i in set(range(self._num_actions)) - {action}:
            self._probs[i] -= self._step_size * self._probs[i] * reward
