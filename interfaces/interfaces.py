
from abc import ABCMeta, abstractmethod


class Environment(metaclass=ABCMeta):

    @property
    @abstractmethod
    def actions(self):
        pass

    @abstractmethod
    def act(self, action):
        pass

    @abstractmethod
    def observation(self):
        pass

    @abstractmethod
    def showObs(self, observaton):
        pass


class Bot(metaclass=ABCMeta):

    @abstractmethod
    def decision(self, observation):
        pass

    @abstractmethod
    def learnFromExp(self, exp):
        pass
