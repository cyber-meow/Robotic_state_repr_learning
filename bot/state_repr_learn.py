
import numpy as np
from sklearn.preprocessing import normalize

from utility import set_all_args


def gradient(f, x, delta):
    """This function is used to compute the gradient of an arbitrary 
    function at a particular point using the definition, x must be an
    1d array.
    But it shouldn't be used here in the gradient descent algorithm
    of the loss function because the complexity would be too high. 
    Instead, we can use is to verifty that the analytic expressions
    given are the right ones.
    """
    res = np.empty(np.shape(x)[0])
    for i in range(np.shape(x)[0]):
        deltaArr = np.zeros(np.shape(x)[0])
        deltaArr[i] = delta
        res[i] = (f(x + deltaArr) - f(x - deltaArr)) / (2 * delta)
    return res

def array_outer(arr1, arr2):
    """array_outer(arr1, arr2) compute the outer product pairwisely for
    the elements of arr1 and arr2
    That is, [np.outer(arr1[i],arr2[i]) for i in range(len(arr1))]
    """
    return np.einsum("...i,...j->...ij", arr1, arr2)

def div0(a,b):
    with np.errstate(divide='ignore', invalid='ignore'):
        c = np.true_divide(a, b)
        c = np.nan_to_num(c)
    return c


class StateReprLearn(object):

    # internal parameters
    epsilon = 0.1
    error = 1e-4
    k = 100  # loss function for only samples at most k steps apart
    p = 0.1  # the probability that epsilon gets larger for each tour

    # experimental data is a triple of arrays (o_ts, a_ts, r_ts)
    def __init__(self, obser_dim, st_dim, data, **kwargs):
        self._obser_dim = obser_dim
        self._st_dim = st_dim
        self.data = data
        # this is in fact the transpose of W
        self.W = np.random.random((obser_dim, st_dim))
        set_all_args(self, kwargs)

    @property
    def obser_dim(self):
        return self._obser_dim

    @property
    def st_dim(self):
        return self._st_dim

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

    @property
    def states(self):
        return self._states


    def pre_compute_obs(self):

        """The function must be called every time the observations
        are updated.  It's used to compute some parameters that 
        will be repeatedly used later.
        -----------------------------------------------------------------
        - self._obs: 2d array, array of observations
        - self._obs_delta: 2d array, array of đo_t = o_(t+1) - o_(t)
        - self._same_action: array of couples t1, t2 such that a_t1 = at_2
        - self._sanr: int, 
          count of t1, t2 such that a_t1 = a_t2 and r_(t1+1) != r_(t2+1)
        - self._obs1_delta: 2d array,
          values of đo_t but only with t1s of _same_action
        - self._obs2_delta: 2d array,
          values of đo_t but only with t2s of _same_action
        - self._obs_diff1: 2d array,
          array of o_t2 - o_t1 for each couple (t1,t2) from _same_action
        - self._obs_diff2: 2d array,
          just like _obs_diff1, but with the addition condition 
          r_(t1+1) != r_(t2+1)
        - self._obs_delta_diff: 2d array,
          array of đo_t2 - đo_t1 for each couple (t1,t2) from _same_action
        """
        
        # in general
        self._obs = np.array(self.data[0])
        self._obs_delta = self._obs[1:] - self._obs[:-1]
        # a_t1 = a_t2, with eventually r_(t1+1) != r_(t2+1)
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
        self._states = np.dot(self._obs, self.W)
        self._st_delta = self._states[1:] - self._states[:-1]
        self._st_delta_norm = np.linalg.norm(self._st_delta, axis=1)
        # a_t1 = a_t2, with eventually r_t1 != r_t2
        sa = len(self._same_action)
        self._st1_delta_nor = np.empty((sa, self.st_dim))
        self._st2_delta_nor = np.empty((sa, self.st_dim))
        self._st1_delta_norm = np.empty(sa)
        self._st2_delta_norm = np.empty(sa)
        self._st_diff_nor1 = np.empty((sa, self.st_dim))
        self._st_diff_nor2 = np.empty((self._sanr, self.st_dim))
        self._st_diff_norm1 = np.empty(sa)
        self._st_diff_norm2 = np.empty(self._sanr)
        self._st_expdiff1 = np.empty(sa)
        self._st_expdiff2 = np.empty(self._sanr)
        self._st_delta_diff = np.empty((sa, self.st_dim))
        j = 0
        for i, (t1, t2, b) in enumerate(self._same_action):
            self._st1_delta_nor[i] = normalize(self._st_delta[t1, None])
            self._st2_delta_nor[i] = normalize(self._st_delta[t2, None])
            self._st1_delta_norm[i] = self._st_delta_norm[t1]
            self._st2_delta_norm[i] = self._st_delta_norm[t2]
            st_diff = self._states[t2] - self._states[t1]
            st_diff_nor = normalize(st_diff[None,:])
            st_diff_norm = np.linalg.norm(st_diff)
            st_expdiff = np.exp(-st_diff_norm)
            self._st_diff_nor1[i] = st_diff_nor
            self._st_diff_norm1[i] = st_diff_norm
            self._st_expdiff1[i] = st_expdiff
            self._st_delta_diff[i] = (self._st_delta[t2] - self._st_delta[t1])
            if not b:
                self._st_diff_nor2[j] = st_diff_nor
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

    # note that it's the transpose
    def gradient(self):
        temp_grad = (
            2 * array_outer(self._obs_delta, self._st_delta)).mean(axis=0)
        prop_grad = (
            2 * (self._st2_delta_norm - self._st1_delta_norm)[:,None,None]
            * (array_outer(self._obs2_delta, self._st2_delta_nor)
            - array_outer(self._obs1_delta, self._st1_delta_nor))).mean(axis=0)
        cau_grad = (
            - self._st_expdiff2[:,None,None]
            * array_outer(self._obs_diff2, self._st_diff_nor2)).mean(axis=0)
        rep_grad = (
            - self._st_expdiff1[:,None,None]
            * array_outer(self._obs_diff1, self._st_diff_nor1)
            * (self._st_delta_diff_norm**2)[:,None,None]
            + self._st_expdiff1[:,None,None]
            * 2 * array_outer(self._obs_delta_diff, self._st_delta_diff)
           ).mean(axis=0)
        return temp_grad + prop_grad + cau_grad + rep_grad
        
    def gradient_descent_step(self):
        grad = self.gradient()
        if np.linalg.norm(grad) < self.error:
            return False
        W_copy = self.W.copy()
        loss_copy = self.loss_func()
        if np.random.random() < self.p:
            print('epsilon gets larger')
            self.epsilon *= 1.2
        self.W -= self.epsilon * grad
        if self.loss_func() > loss_copy:
            print('epsilon gets smaller')
            self.W = W_copy
            self.epsilon *= 0.7
        return True

    def gradient_descent(self, maxiter=1000):
        for _ in range(maxiter):
            if not self.gradient_descent_step():
                break
