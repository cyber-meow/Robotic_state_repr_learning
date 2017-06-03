
import numpy as np
from sklearn.decomposition import PCA
from sklearn.externals import joblib

from inter.interfaces import Bot
from bot.state_repr_learn import StateReprLearn
from utility import set_all_args


class QLBot(Bot):

    cycle = 500
    qlfit_max_iter = 200
    qlfit_intra_step = 50

    def __init__(self, q_learning, st_dim, **kwargs): 
        """st_dim is not used here, just for signature consitency"""
        self.q_learning = q_learning
        self._data = [[], [], []]
        self._iter_num = 0
        set_all_args(self, kwargs)

    @property
    def actions(self):
        return self._actions

    @actions.setter
    def actions(self, actions):
        self._actions = actions
        self.q_learning.actions = actions

    @property
    def data(self):
        return self._data

    def compute_state(self, observation):
        """
        observation can be just one observation or a list of obervations
        in the two cases we need to either return just a state or a list 
        of states.

        Here we use raw observations.
        """
        return observation

    def decision(self, observation):
        if self._iter_num < self.cycle:
            return self.actions[np.random.randint(len(self.actions))]
        state = self.compute_state(observation)
        return self.q_learning.decision(state)

    def learn_from_experience(self, exp):
        for i in range(3):
            self._data[i].append(exp[i])
        self._iter_num += 1
        if self._iter_num % self.cycle == 0:
            states = self.compute_state(self.data[0])
            self.q_learning.fit(
                [states, self.data[1], self.data[2]], 
                self.qlfit_max_iter, self.qlfit_intra_step)

    def save(self, pathname):
        """
        better store library values because the implementation 
        will not be modified within the same version
        """
        joblib.dump(self.q_learning.mlp, "{}_NFQ_MLP.pkl".format(pathname))

    def retrive(self, pathname):
        self.q_learning.mlp = joblib.load("{}_NFQ_MLP.pkl".format(pathname))


class QLBotSRL(QLBot):
    """use the states learned by the main algorithm"""

    def __init__(self, q_learning, st_dim, **kwargs):
        super().__init__(q_learning, st_dim, **kwargs)
        self.st_dim = st_dim

    def compute_state(self, observation):
        return np.dot(observation, self.srl.W)

    def learn_from_experience(self, exp):
        for i in range(3):
            self._data[i].append(exp[i])
        self._iter_num += 1
        if self._iter_num % self.cycle == 0:
            if self._iter_num == self.cycle:
                obser_dim = len(self.data[0][0])
                self.srl = StateReprLearn(obser_dim, self.st_dim, self.data)
            else:
                self.srl.data = self.data
                self.srl.pre_compute_states()
            print("start srl gradient descent")
            self.srl.gradient_descent(300)
            print("srl gradient descent finished, now Q-learning")
            self.q_learning.fit(
                [self.srl.states, self.data[1], self.data[2]],
                self.qlfit_max_iter, self.qlfit_intra_step)

    def save(self, pathname):
        joblib.dump(self.q_learning.mlp, "{}_NFQ_MLP.pkl".format(pathname))
        joblib.dump(self.srl.W, "{}_SRL_W.pkl".format(pathname))

    def retrive(self, pathname):
        self.q_learning.mlp = joblib.load("{}_NFQ_MLP.pkl".format(pathname))
        self.srl.W = joblib.load("{}_SRL_W.pkl".format(pathname))


class QLBotPCA(QLBot):
    """use the first five principal components"""

    def __init__(self, q_learning, st_dim, **kwargs):
        super().__init__(q_learning, st_dim, **kwargs)
        self.pca = PCA(n_components=st_dim)

    def compute_state(self, observation):
        """this only works for one observation"""
        return self.pca.transform([observation])[0]

    def learn_from_experience(self, exp):
        for i in range(3):
            self._data[i].append(exp[i])
        self._iter_num += 1
        if self._iter_num % self.cycle == 0:
            states = self.pca.fit_transform(self.data[0])
            print("start Q-learning")
            self.q_learning.fit(
                [states, self.data[1], self.data[2]],
                self.qlfit_max_iter, self.qlfit_intra_step)
    
    def save(self, pathname):
        joblib.dump(self.q_learning.mlp, "{}_NFQ_MLP.pkl".format(pathname))
        joblib.dump(self.pca, "{}_PCA.pkl".format(pathname))

    def retrive(self, pathname):
        self.q_learning.mlp = joblib.load("{}_NFQ_MLP.pkl".format(pathname))
        self.pca = joblib.load("{}_SRL_PCA.pkl".format(pathname))
