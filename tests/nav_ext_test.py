
import sys
import os
sys.path.insert(0, os.path.abspath('..'))

import matplotlib.pyplot as plt
import matplotlib.animation as animation

from inter.interaction import Interaction
from environment.nav_env_ext import NavEnvExt
from bot.explore_bot import ExploreBot


nav_env = NavEnvExt()
bot = ExploreBot()
inter = Interaction(nav_env, bot)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

obs1 = nav_env.show_observation(nav_env.top_down_view(100))
obs2 = nav_env.show_observation(inter._observation)

im1 = ax1.imshow(obs1, interpolation='none', origin='lower')
im2 = ax2.imshow(obs2, interpolation='none', origin='lower')

def animate(*args):
    inter.interact()
    im1.set_array(
        nav_env.top_down_view(100))
    im2.set_array(
        nav_env.show_observation(inter._observation))
    return im1, im2

def run():
    ani = animation.FuncAnimation(fig, animate, 1000, interval=100, blit=True)
    #ani.save("nav_env.mp4", writer='ffmpeg')
    plt.show()
    return ani
