
import sys
import os
sys.path.insert(0, os.path.abspath('..'))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

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
        self.test_performance()

    def test_performance(self):
        avg, std = self.inter_test.compute_avg_reward(20, 50)
        print("test the performance of the robot")
        self.avg_rewards.append(avg)
        self.reward_stds.append(std)
    
    def run_one_cycle(self):
        self.inter.env.__init__()
        self.inter.interact_serie(500)
        self.test_performance()
        

# the bot knows the internal states, this shall be the upper bound
#bot1 = QLBotTest(NavEnvExtSpe, QLBot)

# using learned states by the main algorithm
#bot2 = QLBotTest(NavEnvExt, QLBotSRL)

# using five first principal components
#bot3 = QLBotTest(NavEnvExt, QLBotPCA)

# using raw observations
#bot4 = QLBotTest(NavEnvExt, QLBot)


class ShowAnimation(object):
    
    def __init__(self, inter):
        self.inter = inter
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(16, 6))
        self.ax1.axis('off'); self.ax2.axis('off')

        obs1 = inter.env.top_down_view(50)
        obs2 = inter.env.show_observation(inter._observation)

        self.im1 = self.ax1.imshow(obs1, origin='lower')
        self.im2 = self.ax2.imshow(obs2, interpolation='none', origin='lower')

    def animate(self, *args):
        self.inter.interact_no_learn()
        self.im1.set_array(self.inter.env.top_down_view(50))
        self.im2.set_array(
            self.inter.env.show_observation(self.inter._observation))
        return self.im1, self.im2

    def run(self, name=None, length=500):
        ani = animation.FuncAnimation(
            self.fig, self.animate, length, interval=100, blit=True)
        if name is not None:
            ani.save(name, writer='ffmpeg')
        plt.show()
        return ani


def plot_with_std(d1, d2, d3, d4, n, num_steps):
    
    labels = ['RL on robot pose', 'RL on learned states',
              'RL on principal components', 'RL on observations']
    line_colors = ['black', 'forestgreen', 'darkviolet', 'peru']
    suface_colors = ['grey', 'lightgreen', 'slateblue', 'sandybrown']
    
    x_axis = np.linspace(0, num_steps, n)
    
    for i, d in enumerate([d1, d2, d3, d4]):
        avg, std = np.array(d[0][:n]), np.array(d[1][:n])
        plt.plot(x_axis, avg, color=line_colors[i])
        plt.fill_between(x_axis, avg-std, avg+std, label=labels[i],
                         facecolor=suface_colors[i], alpha=0.3)
    plt.xlabel("Number of training steps")
    plt.ylabel("Reward per 50 step episode")
    lgd = plt.legend(loc=9, bbox_to_anchor=(0.46, 1.17), ncol=2)
    plt.savefig("q_learning_curves_mine", 
                bbox_extra_artist=(lgd,), bbox_inches='tight')
    plt.show()

