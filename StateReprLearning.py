
import numpy as np
from Utility import setAllArgs


# compute the gradient of an arbitrary function at a particular point
# using the definitoin, x must be an 1d array
def gradient(f, x, delta):
    res = np.empty(np.shape(x)[0])
    for i in range(np.shape(x)[0]):
        deltaArr = np.zeros(np.shape(x)[0])
        deltaArr[i] = delta
        res[i] = (f(x + deltaArr) - f(x - deltaArr)) / (2 * delta)
    return res

# array_outer(arr1, arr2) = [np.outer(arr1[i],arr2[i]) for i in ...]
def array_outer(arr1, arr2):
    return np.einsum("...i,...j->...ij", arr1, arr2)


class StateReprLearn(object):

    # internal parameters
    epsilon = 0.1
    error = 1e-4
    k = 100

    # experimental data is a triple of arrays (o_ts, a_ts, r_ts)
    def __init__(self, obser_dim, st_dim, data, **kwargs):
        self.obser_dim = obser_dim
        self.st_dim = st_dim
        self.data = data
        self.W = np.random.random((st_dim, obser_dim))
        setAllArgs(self, kwargs)

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data):
        self._data = data
        self.pre_compute_obs()

    @property
    def states(self):
        return self._states

    @property
    def W(self):
        return self._W

    @W.setter
    def W(self, W):
        self._W = W
        self.pre_compute_states()

    def pre_compute_obs(self):
        # in general
        self._obs = self.data[0]
        self._obs_delta = self._obs[1:] - self._obs[:-1]
        # a_t1 = a_t2, with eventually r_t1 != r_t2
        self._same_action = []
        self._sanr = 0
        obs1_delta, obs2_delta = [], []
        obs_diff1, obs_diff2 = [], []
        obs_delta_diff = []
        for t1 in range(len(self._obs)-2):
            for t2 in range(t1+1, min(t1+self.k+1, len(self._obs)-1)):
                if self.data[1][t1] == self.data[1][t2]:
                    obs1_delta.append(self._obs_delta[t1])
                    obs2_delta.append(self._obs_delta[t2])
                    obs_diff1.append(self._obs[t2] - self._obs[t1])
                    obs_delta_diff.append(
                        self._obs_delta[t2] - self._obs_delta[t1])
                    if self.data[2][t1+1] != self.data[2][t2+1]:
                        self._same_action.append((t1, t2, False))
                        obs_diff2.append(obs_diff1[-1])
                        self._sanr += 1
                    else:
                        self._same_action.append((t1, t2, True))
        self._obs1_delta = np.array(obs1_delta)
        self._obs2_delta = np.array(obs2_delta)
        self._obs_diff1 = np.array(obs_diff1)
        self._obs_diff2 = np.array(obs_diff2)
        self._obs_delta_diff = np.array(obs_delta_diff)

    def pre_compute_states(self):
        # in general
        self._states = np.dot(self._obs, self.W.transpose())
        self._st_delta = self._states[1:] - self._states[:-1]
        self._st_delta_norm = np.linalg.norm(self._st_delta, axis=1)
        # a_t1 = a_t2, with eventually r_t1 != r_t2
        sa = len(self._same_action)
        self._st1_delta = np.empty((sa, self.st_dim))
        self._st2_delta = np.empty((sa, self.st_dim))
        self._st1_delta_norm = np.empty(sa)
        self._st2_delta_norm = np.empty(sa)
        self._st_diff1 = np.empty((sa, self.st_dim))
        self._st_diff2 = np.empty((self._sanr, self.st_dim))
        self._st_diff_norm1 = np.empty(sa)
        self._st_diff_norm2 = np.empty(self._sanr)
        self._st_expdiff1 = np.empty(sa)
        self._st_expdiff2 = np.empty(self._sanr)
        self._st_delta_diff = np.empty((sa, self.st_dim))
        j = 0
        for i, (t1, t2, b) in enumerate(self._same_action):
            self._st1_delta[i] = self._st_delta[t1]
            self._st2_delta[i] = self._st_delta[t2]
            self._st1_delta_norm[i] = self._st_delta_norm[t1]
            self._st2_delta_norm[i] = self._st_delta_norm[t2]
            st_diff = self._states[t2] - self._states[t1]
            st_diff_norm = np.linalg.norm(st_diff)
            st_expdiff = np.exp(-st_diff_norm)
            self._st_diff1[i] = st_diff
            self._st_diff_norm1[i] = st_diff_norm
            self._st_expdiff1[i] = st_expdiff
            self._st_delta_diff[i] = (self._st_delta[t2] - self._st_delta[t1])
            if not b:
                self._st_diff2[j] = st_diff
                self._st_diff_norm2[j] = st_diff_norm
                self._st_expdiff2[j] = st_expdiff
                j += 1
        assert j == self._sanr
        self._st_delta_diff_norm = np.linalg.norm(self._st_delta_diff, axis=1)
        
    def loss_func(self):
        temp_loss = (self._st_delta_norm**2).mean()
        prop_loss = ((self._st2_delta_norm - self._st1_delta_norm)**2).mean()
        cau_loss = self._st_expdiff2.mean()
        rep_loss = (self._st_expdiff1 * self._st_delta_diff_norm**2).mean()
        return temp_loss + prop_loss + cau_loss +rep_loss

    def gradient(self):
        temp_grad = (
            2 * array_outer(self._st_delta, self._obs_delta)).mean(axis=0)
        prop_grad = (
            2 * (self._st2_delta_norm - self._st1_delta_norm)[:,None,None]
            * (array_outer(self._st2_delta, self._obs2_delta)
            / self._st2_delta_norm[:,None,None]
            - array_outer(self._st1_delta, self._obs1_delta)
            / self._st1_delta_norm[:,None,None])).mean(axis=0)
        cau_grad = (
            - self._st_expdiff2[:,None,None]
            * array_outer(self._st_diff2, self._obs_diff2)
            / self._st_diff_norm2[:,None,None]).mean(axis=0)
        rep_grad = ((
            - self._st_expdiff1[:,None,None]
            * array_outer(self._st_diff1, self._obs_diff1)
            / self._st_diff_norm1[:,None,None]
            * (self._st_delta_diff_norm**2)[:,None,None])
            + self._st_expdiff1[:,None,None]
            * 2 * array_outer(self._st_delta_diff, self._obs_delta_diff)
            ).mean(axis=0)
        return temp_grad + prop_grad + cau_grad + rep_grad
        
    """
    def gradientDescent(self, maxiter = 1000):
        for _ in range(maxiter):
            grad = gradient(self.lossFunc, self.W, self.delta)
    """


data = np.array([[5,3,2],[7,4,6],[2,2,1],[1,1,1]]), [10,10,10,10], [3,2,3,3]
srl = StateReprLearn(3,2,data)

def srl_loss(W):
    srl.W = W.reshape(2,3)
    return srl.loss_func()
