#Q learning for markov decision process
#Ting-Yao Hu, 2015.08

import sys

class MyMDP(object):

    ### Q[s,a] = Q[s,a]*(1-alpha) + (r+gamma*max_a'Q[s',a'])*alpha
    def __init__(self, alpha = 0.5, gamma = 0.8, iternum = 1000):
        self.alpha = alpha
        self.gamma = gamma
        self.iternum = iternum

    def init_from_lsts(self, statelst, actionlst):
        self.state2idx = dict([(state,idx) for idx, state in enumerate(statelst)])
        self.action2idx = dict([(act,idx) for idx, act in enumerate(actionlst)])
        self.actionlst = actionlst
        self.statelst = statelst

        act_num = len(self.actionlst)
        state_num = len(self.statelst)
        self.Qfunc = [ [0]*act_num for idx in range(state_num) ]


    def init_from_history(self, history):
        statelst = [ his[0] for his in history]
        statelst2 = [ his[3] for his in history]
        actionlst = [ his[1] for his in history]

        stateset = set(statelst+statelst2)
        if None in stateset: stateset.remove(None)
        actionset = set(actionlst)
        self.init_from_lsts(list(stateset),list(actionset))

    def map2idx(self, history):
        history_idxs = []
        for idx in range(len(history)):
            if history[idx][3] is None: tup = (self.state2idx[history[idx][0]], self.action2idx[history[idx][1]],history[idx][2],None)
            else: tup = (self.state2idx[history[idx][0]], self.action2idx[history[idx][1]],history[idx][2],self.state2idx[history[idx][3]])
            history_idxs.append(tup)
        return history_idxs

    def q_learn(self, history):
        his_len = len(history)
        history_idxs = self.map2idx(history)
        for it in range(self.iternum):
            for idx in range(his_len):
                stateidx = history_idxs[idx][0]
                actidx = history_idxs[idx][1]
                reward = history_idxs[idx][2]
                stateidx2 = history_idxs[idx][3]
                if stateidx2 is None:
                    self.Qfunc[stateidx][actidx] = \
                        self.Qfunc[stateidx][actidx]*(1-self.alpha) + reward*self.alpha
                else:
                    self.Qfunc[stateidx][actidx] = \
                        self.Qfunc[stateidx][actidx]*(1-self.alpha) + \
                        (reward+self.gamma*max(self.Qfunc[stateidx2]))*self.alpha

    def policy(self,state,distance_func=None):
        try:
            stateidx = self.state2idx[state]
        except KeyError:
            nearest = self.nearest_state(state,distance_func)
            stateidx = self.state2idx[nearest]
        actidx = self.Qfunc[stateidx].index(max(self.Qfunc[stateidx]))
        return self.actionlst[actidx]

    def print_policy(self):
        for idx, state in enumerate(self.statelst):
            print idx, '. state: ', state, ' action: ', self.policy(state)

    def nearest_state(self,state,distance_func):
        min_distance = -1
        for idx, state2 in enumerate(self.statelst):
            distance = distance_func(state,state2)
            if min_distance==-1 or min_distance>distance:
                nearest = state2
                min_distance = distance
        return nearest

def TestFunc():
    print 'test'
    mdp = MyMDP()

if __name__=='__main__':
    TestFunc()
