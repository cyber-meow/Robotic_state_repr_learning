
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from inter.interfaces import Environment, Bot


class Interaction(object):
    """
    This is rather for test purposes once the robot has learned something
    from its experience.
    For the main interaction class please look at Interaction.
    """

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

    def interact_no_learn(self):
        self._env_state_his.append(self.env.state)
        action = self.bot.decision(self._observation)
        exp = self._observation, action, self._reward
        self._observation, self._reward = self.env.act(action)
        return exp

    def interact(self):
        exp = self.interact_no_learn()
        self.bot.learn_from_experience(exp)
    
    def interact_serie(self, iter_num):
        for _ in range(iter_num):
            self.interact()

    def observation_serie(self, print_reward=True, save=False, path=None):
        fig, ax = plt.subplots()
        ax.axis('off')
        obs = self.env.show_observation(self._observation)
        im = plt.imshow(obs, interpolation='none', origin='lower')
        def animate(*args):
            self.interact_no_learn()
            if print_reward:
                print("reward: {}".format(self._reward))
            im.set_array(self.env.show_observation(self._observation))
            return im,
        ani = animation.FuncAnimation(
            fig, animate, 200, interval=120, blit=True)
        if save:
            assert path is not None
            ani.save(path, writer='ffmpeg')
        plt.show()
        return ani

    def compute_avg_reward(self, num_episode, num_step):
        rewards = []
        for _ in range(num_episode):
            self.env.__init__()
            reward_sum = 0
            for _ in range(num_step):
                self.interact_no_learn()
                reward_sum += self._reward
            rewards.append(reward_sum)
        return np.mean(rewards)
        
