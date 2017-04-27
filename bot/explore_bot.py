
import numpy as np

from interfaces.interfaces import Bot
from utility import set_all_args


class ExploreBot(Bot):

    k = 100

    def __init__(self, **kwargs):
        self.actions = None
        # triple of lists of o_t, a_t, r_t
        self._data = [[], [], []]
        set_all_args(self, kwargs)

    @property
    def data(self):
        return self._data

    def decision(self, observation):
        if len(self.data[0]) >= self.k:
            if np.random.random() < 1/2:
                return self.data[1][-self.k]
        return self.actions[np.random.randint(len(self.actions))]
        
    def learn_from_experience(self, exp):
        for i in range(3):
           self._data[i].append(exp[i])
