#Linear Function Approximation for Q-learning of MDP
#Ting-Yao Hu, 2015.09

import sys
import numpy as np

class FA_MDP(object):
    def __init__(self, alpha = 0.01, gamma = 0.8, iternum = 20):
        self.alpha = alpha
        self.gamma = gamma
        self.iternum = iternum

    def Init_From_Lsts(self, actionlst, feat_dim):
        self.action2idx = dict([(act,idx) for idx, act in enumerate(actionlst)])
        self.actionlst = actionlst
        actnum = len(self.actionlst)
        self.theta = np.zeros((actnum,feat_dim),dtype='float32')

    def Init_From_History(self, history):
        feat_dim = history[0][0].shape[0]
        actionlst = [ his[1] for his in history]
        actionset = set(actionlst)

        actionset = set(actionlst)
        self.Init_From_Lsts(list(actionset),feat_dim)

    def Map2Idx(self, history):
        history_idxs = []
        for idx in range(len(history)):
            if history[idx][3] is None: tup = (history[idx][0], self.action2idx[history[idx][1]],history[idx][2],None)
            else: tup = (history[idx][0], self.action2idx[history[idx][1]],history[idx][2],history[idx][3])
            history_idxs.append(tup)
        return history_idxs


    def QLearn(self, history):
        his_len = len(history)
        history_idxs = self.Map2Idx(history)
        for it in range(self.iternum):
            for idx in range(his_len):
                statevec = history_idxs[idx][0]
                actidx = history_idxs[idx][1]
                reward = history_idxs[idx][2]
                state2vec = history_idxs[idx][3]

                if state2vec is None:
                    self.theta[actidx] += self.alpha*(reward - self.theta[actidx].dot(statevec))
                else:
                    self.theta[actidx] += statevec*self.alpha*(reward + self.gamma*max(self.theta.dot(state2vec)) - self.theta[actidx].dot(statevec))


    def Policy(self, state_vec):
        actidx = np.argmax(self.theta.dot(state_vec))
        return self.actionlst[actidx]

    #print theta here
    def PrintPolicy(self):
        print 'Action list: ',self.actionlst
        print 'Theta matrix: ',self.theta

def TestFunc():
    print 'test'
    mdp = FA_MDP()
