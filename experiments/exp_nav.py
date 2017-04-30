
import sys
import os
sys.path.insert(0, os.path.abspath('..'))

import numpy as np

from inter.interaction import Interaction
from environment.nav_env_simp import NavEnv
from bot.explore_bot import ExploreBot
from bot.state_repr_learn import StateReprLearn, gradient
from plot_exp_nav import plot_states


nav_env = NavEnv()
bot = ExploreBot()
inter = Interaction(nav_env, bot)
# inter.interact_serie(5000)

# bot_srl = StateReprLearn(300, 2, bot.data)
# try to learn the representation
# bot_srl.gradient_descent(100)

# plot_states(np.array(inter.env_state_his), bot_srl.states, 'x')
# plot_states(np.array(inter.env_state_his), bot_srl.states, 'y')
