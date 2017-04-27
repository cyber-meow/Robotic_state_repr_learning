
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from interfaces.interfaces import Environment, Bot


class Interaction(object):

    def __init__(self, env, bot):
        assert isinstance(env, Environment)
        assert isinstance(bot, Bot)
        self.env = env
        self.bot = bot

    @property
    def bot(self):
        return self._bot

    @bot.setter
    def bot(self, bot):
        self._bot = bot
        self._bot.actions = self.env.actions
        self._reward = 0

    @property
    def env(self):
        return self._env

    @env.setter
    def env(self, env):
        self._env = env
        self._observation = env.observation()
        self._env_state_his = []

    @property
    def env_state_his(self):
        return self._env_state_his

    def interact(self):
        self._env_state_his.append(self.env.state)
        action = self.bot.decision(self._observation)
        exp = self._observation, action, self._reward
        self.bot.learn_from_experience(exp)
        self._observation, self._reward = self.env.act(action)
    
    def interact_serie(self, iter_num):
        for _ in range(iter_num):
            self.interact()

    def observation_serie(self):
        fig, ax = plt.subplots()
        obs = self.env.show_observation(self._observation)
        im = plt.imshow(obs, interpolation='none', origin='lower')
        def animate(*args):
            self.interact()
            im.set_array(self.env.show_observation(self._observation))
            return im,
        ani = animation.FuncAnimation(fig, animate, interval=50, blit=True)
        plt.show()
