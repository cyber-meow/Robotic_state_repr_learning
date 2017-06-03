
import sys
import os
sys.path.insert(0, os.path.abspath('..'))

from inter.interaction import Interaction
from environment.nav_env_ext import NavEnvExtSpe
from bot.q_learning_bot import QLBot
from bot.NFQ import NFQ


nav_env = NavEnvExtSpe()
bot = QLBot(NFQ())
inter = Interaction(nav_env, bot)
inter_test = Interaction(nav_env, bot)

avg_rewards = []

def run_one_cycle():
    inter.interact_serie(500)
    avg_rewards.append(inter_test.compute_avg_reward(20, 50))


