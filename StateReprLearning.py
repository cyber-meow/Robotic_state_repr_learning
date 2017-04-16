
import numpy as np
from Utility import setAllArgs


# compute the gradient of a function at a particular point
def gradient(f, x, delta):
    res = np.empty(np.shape(x)[0])
    for i in np.shape(x)[0]:
        deltaArr = np.zeros(np.shape(x)[0])
        deltaArr[i] = 1
        res[i] = (f(x + deltaArr) - f(x - deltaArr)) / (2 * delta)
    return res


class StateReprLearn(object):

    # parameters for gradient descent
    epsilon = 0.1
    delta = 1e-6
    error = 1e-4

    # experimental data D is an array of (o_t, a_t, r_t) in order
    def __init__(self, obserDim, stDim, **kwargs):
        self.data = np.empty(0)
        self.W = np.random.random(stDim, obserDim)
        setAllArgs(self, kwargs)

    # complexity: len(self.data) * obserDim * stDim (k constant)
    def lossFunc(self, W, k):
        states = np.dot(W, self.data[:,0])
        stChs = states[1:] - states[:-1]
        stDiffs = np.linalg.norm(stChs, axis = 1)
        tempLoss = mean(stDiffs**2)
        N, propLoss, cauLoss, repLoss = 0, 0, 0, 0
        for i in range(len(self.data)-2):
            for j in range(i+1, min(i+k+1, len(self.data)-1)):
                N += 1
                if self.data[i][1] == self.data[j][1]:
                    propLoss += (stDiffs[i] - stDiffs[j])**2
                    stDiff = np.linalg.norm(states[j]-state[i])
                    stSim = np.exp(-stDiff)
                    stChDiff = np.linalg.norm(stChs[j] - stChs[i])
                    repLoss += stDim * stChDiff ** 2
                    if self.data[i][2] != self.data[j][2]:
                        cauLoss += stSim
        propLoss /= N; cauLoss /= N; repLoss /= N
        return tempLoss + propLoss + cauLoss +repLoss

    def gradientDescent(self, maxiter = 1000):
        for _ in range(maxiter):
            grad = gradient(self.lossFunc, self.W, self.delta)




