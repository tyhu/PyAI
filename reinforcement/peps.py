"""
Parameter Exploration Policy Search
Ting-Yao Hu, 2017.10
"""

import numpy as np

class Gause_PEPS(object):
    def __init__(self, mu, sigma=0.01):
        self.mu = mu
        self.sigma = np.ones_like(mu)*sigma
        self.baseline_dict = {}
        self.baseline = None

    def sample_parameters(self):
        return np.random.normal(mu,sigma)

    def update_with_global_baseline(self, thetas, rewards, lr=0.001, gamma=0.1):
        if self.baseline is None: self.baseline = np.mean(rewards)

        dmu, dsig = 0,0
        for theta, reward in zip(thetas, rewards):
            dmu += (theta-self.mu)*(reward-self.baseline)
            dsig += (np.square(theta-self.mu)-np.square(self.sigma))/self.sigma
        self.mu = self.mu + lr*dmu
        self.sigma = self.sigma + lr*dsig

        self.baseline = gamma*np.mean(rewards)+(1-gamma)*self.baseline

    #def update_with_local_baseline(self, thetas, rewards, baseline_idxs):
