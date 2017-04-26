
import numpy as np

from interfaces.interfaces import Bot
from utility import setAllArgs


class ExploreBot(Bot):

    k = 100

    def __init__(self, **kwargs):
        self.actions = None
        # triple of lists of o_t, a_t, r_t
        self.data = [[], [], []]
        setAllArgs(self, kwargs)

    def decision(self, observation):
        if len(self.data[0]) >= self.k:
            if np.random.random() < 1/2:
                return self.data[1][-self.k]
        return self.actions[np.random.randint(len(self.actions))]
        
    def learnFromExp(self, exp):
        for i in range(3):
           self.data[i].append(exp[i])
