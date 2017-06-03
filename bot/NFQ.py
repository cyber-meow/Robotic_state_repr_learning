
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.exceptions import NotFittedError

from utility import set_all_args 


class NFQ(object):

    gamma = 0.9
    beta = 0.8

    def __init__(self, **kwargs):
        self.mlp = MLPRegressor(
            hidden_layer_sizes=(5,5), activation='logistic', batch_size=400)
        set_all_args(self, kwargs)
        

    def fit(self, data, max_iter=300, intra_step=10):
        """
        data is the triple (ss, as, rs)
        """
        for _ in range(max_iter):
            inputs, targets = self.compute_inputs_targets(data)
            for _ in range(intra_step):
                self.mlp.partial_fit(inputs, targets)


    def compute_inputs_targets(self, data):
         
        inputs, targets = [], []
        
        for i in range(len(data[0])-1):
            s, a, r = list(data[0][i]), data[1][i], data[2][i]
            s_next = list(data[0][i+1])
            inputs.append(s + [self.actions.index(a)])
            to_prs = [s_next + [act] for act in range(len(self.actions))]
            try:
                q_values = self.mlp.predict(to_prs)
                targets.append(r + self.gamma * np.max(q_values))
            except NotFittedError:
                targets.append(r)
        
        return np.array(inputs), np.array(targets)
    

    def score(self, data):
        inputs, targes = self.compute_inputs_targets(data)
        return self.mlp.score(inputs, targes)

    
    def decision(self, state):
        state = list(state)
        to_prs = [state + [act] for act in range(len(self.actions))]
        q_values = self.mlp.predict(to_prs)
        ps = np.exp(self.beta * q_values)
        a_num = np.random.choice(len(self.actions), p=ps/np.sum(ps))
        return self.actions[a_num]



