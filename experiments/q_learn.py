
import sys
import os
sys.path.insert(0, os.path.abspath('..'))

from inter.interaction import Interaction
from environment.nav_env_ext import NavEnvExt, NavEnvExtSpe
from bot.q_learning_bot import QLBot, QLBotSRL, QLBotPCA
from bot.NFQ import NFQ


class QLBotTest(object):

    def __init__(self, env, bot):
        assert issubclass(env, NavEnvExt)
        assert issubclass(bot, QLBot)
        self.nav_env = env()
        self.bot = bot(NFQ(), 5)
        self.inter = Interaction(self.nav_env, self.bot)
        self.inter_test = Interaction(self.nav_env, self.bot)
        self.avg_rewards = []
        self.reward_stds = []

    def run_one_cycle(self):
        self.inter.env.__init__()
        self.inter.interact_serie(500)
        avg, std = self.inter_test.compute_avg_reward(20, 50)
        self.avg_rewards.append(avg)
        self.reward_stds.append(std)


# the bot knows the internal states, this shall be the upper bound
bot1 = QLBotTest(NavEnvExtSpe, QLBot)

# using learned states by the main algorithm
bot2 = QLBotTest(NavEnvExt, QLBotSRL)

# using five first principal components
bot3 = QLBotTest(NavEnvExt, QLBotPCA)

# using raw observations
bot4 = QLBotTest(NavEnvExt, QLBot)
