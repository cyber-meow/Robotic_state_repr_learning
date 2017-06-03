
import numpy as np

from inter.interfaces import Bot
from utility import set_all_args


class QLBot(Bot):

    cycle = 500
    qlfit_max_iter = 200
    qlfit_intra_step = 50

    def __init__(self, q_learning, **kwargs): 
        self.q_learning = q_learning
        self._data = [[], [], []]
        self._iter_num = 0
        set_all_args(self, kwargs)

    @property
    def actions(self):
        return self._actions

    @actions.setter
    def actions(self, actions):
        self._actions = actions
        self.q_learning.actions = actions

    @property
    def data(self):
        return self._data

    def compute_state(self, observation):
        """
        observation can be just one observation or a list of obervations
        in the two cases we need to either return just a state or a list 
        of states.

        Here we use raw observations.
        """
        return observation

    def decision(self, observation):
        if self._iter_num < self.cycle:
            return self.actions[np.random.randint(len(self.actions))]
        state = self.compute_state(observation)
        return self.q_learning.decision(state)

    def learn_from_experience(self, exp):
        for i in range(3):
            self._data[i].append(exp[i])
        self._iter_num += 1
        if self._iter_num % self.cycle == 0:
            states = self.compute_state(self.data[0])
            self.q_learning.fit(
                [states, self.data[1], self.data[2]], 
                self.qlfit_max_iter, self.qlfit_intra_step)

