
import sys
import os
sys.path.insert(0, os.path.abspath('..'))
import numpy as np

from inter.interaction import Interaction
from environment.nav_env_simp import NavEnv
from bot.explore_bot import ExploreBot
from bot.state_repr_learn import StateReprLearn, gradient


data = np.array([[5,3,2],[7,4,6],[2,2,1],[1,1,1]]), [10,10,10,10], [3,2,3,3]
srl = StateReprLearn(3,2,data)
print(srl.gradient())

def srl_loss(W):
    srl.W = W.reshape(3,2)
    return srl.loss_func()

# These should give close results
# srl.loss_gradient()
# gradient(srl_loss, srl.W.reshape(6), 1e-6)