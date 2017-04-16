
import numpy as np
from Interfaces import Bot
from Utility import setAllArgs


class ExploreBot(Bot):

    k = 100

    def __init__(self, **kwargs):
        self.actions = None
        # list of (o_t, a_t, r_t)
        self.his = []
        setAllArgs(self, kwargs)

    def decision(self, observation):
        if len(self.his) >= self.k:
            if np.random.random() < 1/2:
                return self.his[-self.k][1]
        return self.actions[np.random.randint(len(self.actions))]
        
    def learnFromExp(self, exp):
        self.his.append(exp)
