
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from Interfaces import Environment, Bot


class Interaction(object):

    def __init__(self, env, bot):
        assert isinstance(env, Environment)
        assert isinstance(bot, Bot)
        self.env = env
        self.bot = bot
        self.bot.actions = env.actions
        self.observation = env.observation()
        self.reward = 0

    def interact(self):
        action = self.bot.decision(self.observation)
        exp = self.observation, action, self.reward
        self.observation, self.reward = self.env.act(action)
        self.bot.learnFromExp(exp)
    
    def observationSerie(self):
        fig, ax = plt.subplots()
        obs = self.env.showObs(self.observation)
        im = plt.imshow(obs, interpolation='none', origin='lower')
        def animate(*args):
            self.interact()
            im.set_array(self.env.showObs(self.observation))
            return im,
        ani = animation.FuncAnimation(fig, animate, interval=50, blit=True)
        plt.show()
