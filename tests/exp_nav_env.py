
import sys
import os
sys.path.insert(0, os.path.abspath('..'))

import numpy as np
import matplotlib.pyplot as plt

from interaction.interaction import Interaction
from env.nav_env_simp import NavEnv
from bot.explore_bot import ExploreBot
from bot.state_repr_learn import StateReprLearn, gradient


nav_env = NavEnv()
bot = ExploreBot()
inter = Interaction(nav_env, bot)
inter.interact_serie(5000)

bot_srl = StateReprLearn(300, 2, bot.data)


def plot_states(realstates, learnedstates):
    xs = realstates[:,0]
    plt.scatter(learnedstates[:,0], learnedstates[:,1], 
                s=5, lw=0, c=xs, vmin=0, vmax=45)
    plt.xlabel("State dimension 1")
    plt.ylabel("State dimension 2")
    plt.colorbar()
    plt.show()

plot_states(np.array(inter.env_state_his), bot_srl.states)

