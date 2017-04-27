
import sys
import os
sys.path.insert(0, os.path.abspath('..'))
import numpy as np

from interaction.interaction import Interaction
from env.nav_env_simp import NavEnv
from bot.explore_bot import ExploreBot
from bot.state_repr_learn import StateReprLearn, gradient


data = np.array([[5,3,2],[7,4,6],[2,2,1],[1,1,1]]), [10,10,10,10], [3,2,3,3]
srl = StateReprLearn(3,2,data)

def srl_loss(W):
    srl.W = W.reshape(2,3)
    return srl.loss_func()

# srl.loss_gradient()
# gradient(srl_loss, srl.W.reshape(6), 1e-6)


nav_env = NavEnv()
bot = ExploreBot()
inter = Interaction(nav_env, bot)
inter.interact_serie(5000)

bot_srl = StateReprLearn(300, 2, bot.data)
