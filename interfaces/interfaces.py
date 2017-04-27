
from abc import ABCMeta, abstractmethod


class Environment(metaclass=ABCMeta):

    @property
    @abstractmethod
    def actions(self):
        """Possible actions that a robot can conduct in this environment.
        Actions must be somthing that can be compared using == in Python,
        for example, one shouldn't represent an action by a numpy array.
        """
        pass

    @abstractmethod
    def act(self, action):
        """Describe the effect of a robot's action"""
        pass

    @property
    @abstractmethod
    def state(self):
        """Represent the actual state of the environment.
        Its real meaning may vary from case to case, what's important is 
        that it can be used to compare with the state representation
        learned by the robot.
        """
        pass

    @abstractmethod
    def observation(self):
        """The observation that the environment can offer to the bot"""
        pass

    @abstractmethod
    def show_observation(self, observaton):
        """An visualization of the observation"""
        pass


class Bot(metaclass=ABCMeta):

    @abstractmethod
    def decision(self, observation):
        """The decision of the robot after observing from the environment"""
        pass

    @abstractmethod
    def learn_from_experience(self, exp):
        """The bot may be able to learn from experience.
        exp: triple of (observation, action, reward)
        """
        pass
