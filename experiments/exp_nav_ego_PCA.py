
import sys
import os
sys.path.insert(0, os.path.abspath('..'))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from inter.interaction import Interaction
from environment.nav_env_ego import NavEnvEgo
from bot.explore_bot import ExploreBot
from bot.state_repr_learn import StateReprLearn
from plot_exp_nav import plot_states


nav_env = NavEnvEgo()
bot = ExploreBot()
inter = Interaction(nav_env, bot)
inter.interact_serie(5000)

bot_srl = StateReprLearn(300, 5, bot.data)
# try to learn the representation
# bot_srl.gradient_descent(100)


pca = PCA()

def fit_pca():
    global state_pcs
    pca.fit(bot_srl.states)
    pca.states = pca.transform(bot_srl.states)

def plot_eigen_values(n):
    plt.bar(range(1,6), pca.explained_variance_, color='turquoise')
    plt.margins(0.08, 0.1)
    plt.ylim(ymin=0)
    plt.ylabel("Eigenvalues of state samples")
    plt.savefig("figures/nav_ego_PCA_{}_evs".format(n))
    plt.show()

# plot the result after n steps of gradient descent
def plot_x(n):
    plot_states(
        np.array(inter.env_state_his), pca.states, 'x', 
        path="figures/nav_ego_PCA_{}_x".format(n), lab="Principal component")
def plot_y(n):
    plot_states(
        np.array(inter.env_state_his), pca.states, 'y',
        path="figures/nav_ego_PCA_{}_y".format(n), lab="Principal component")
