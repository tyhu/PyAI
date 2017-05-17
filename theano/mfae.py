"""
Multi-Factor Auto-Encoder
Ting-Yao Hu, 2017.04
"""

import sys
import numpy as np
import theano
import theano.tensor as T
from util_theano import *
from compact_cnn import *
import cPickle as pickle


def unpoolLayer(X):
    Xup = X.repeat(2,axis=2).repeat(2,axis=3)
    return Xup

def unflattenLayer(X, shape):
    return X.reshape(shape)

"""
multi-factor autoencoder

"""
class MFAE(object):

    def __init__(self,lr=0.001,momentum=0.9,ld2=0.01):
        x = T.tensor4()
        mask = T.matrix()
        o,cost,params = buildMFAE(x,mask)
        updates = updatefunc(cost, params, lr=lr, momentum=momentum)
        self.train = theano.function([x,mask], cost, updates=updates)
        self.cost = theano.function([x,mask],cost)
        self.transform_ = theano.function([x], o)
        self.tparams = params


    def assignParams(self,params):
        for i,tparam in enumerate(self.tparams):
            tparam.set_value(floatX(params[i]))

    def saveParams(self,fn):
        params = []
        for tparam in self.tparams:
            params.append(tparam.get_value())
        pickle.dump(params,open(fn,'wb'))

    def fit_batch(self, X, mask):
        X = floatX(X)
        mask = floatX(mask)
        cost = self.train(X,mask)
        return cost

    def fit(self,Xlst,fac_dims,tiedlst, batchsize=32, iternum=1000):
        for i in range(iternum):
            totalcost = 0
            for x, tied in zip(Xlst,tiedlst):
                x = floatX(x)
                cost = self.train(X, fac_dims, tied)
                totalcost+=cost
            print 'totalcost: ',totalcost

    def transform(self,X):
        X = floatX(X)
        return self.transform_(X)

    def infer_mask(self, Xlst, ld):
        num = Xlst[0].shape[0]
        dim = Xlst[0].shape[1]
        error_vec = np.zeros((dim,))
        for X in Xlst:
            if X.shape[0]!=num: continue
            x1 = X[:num/2]
            x2 = X[num/2:]
            ev = np.sum(((x1-x2)**2),axis=0)/num
            error_vec += ev
        error_vec/=len(Xlst)
        idxs = sorted(range(dim), key=lambda i:error_vec[i], reverse=True)
        mask = np.zeros([dim,1])
        for i in idxs:
            if error_vec[i]<ld: mask[i] = 1
        return mask.T
            

def buildMFAE(x, mask):
    filternum = 40
    rng = np.random.RandomState(12345)
    w_c1 = init_weights_rng((filternum, 3, 5, 5),rng)
    b_c1 = init_weights_rng((filternum,),rng)
    #w_dc1 = w_c1.dimshuffle((1,0,2,3))
    w_dc1 = init_weights_rng((3, filternum, 5, 5),rng)
    b_dc1 = init_weights_rng((3,),rng)

    ### parameters for fully connected factors
    w_h = init_weights_rng((filternum * 16 * 16, 200),rng)
    b_h = init_weights_rng((200,),rng)
    b_dh = init_weights_rng((filternum*16*16,),rng)

    c1 = addSigmoidConvLayer(x,w_c1,b_c1,border_mode=2)
    p1 = addPoolLayer(c1,psize=(2,2))
    flat = p1.flatten(2)
    o = addSigmoidFullLayer(flat,w_h,b_h)    ### hidden representation layer
    h = addSigmoidFullLayer(o,w_h.T,b_dh)
    dflat = unflattenLayer(h, p1.shape)
    dp1 = unpoolLayer(dflat)
    dc1 = addSigmoidConvLayer(dp1,w_dc1,b_dc1,border_mode=2)

    ### calculate disentangled grouping error
    batchsize = x.shape[0]
    masks = mask.repeat(batchsize,axis=0)
    o_masked = o*masks
    o1 = o_masked[:batchsize/2,:]
    o2 = o_masked[batchsize/2:,:]
    cost_g = l2norm([o1-o2])
    
    #ld2 = 0.001
    #ld3 = 0.001
    ld2 = 0.00
    ld3 = 0.01
    datanum = x.shape[0]
    cost = MSE(x,dc1)/datanum+ld2*l2norm([w_c1,w_h,w_dc1])+ld3*cost_g
    params = [w_c1,b_c1,b_dc1,w_h,b_h,b_dh]
    return o, cost, params

"""
multiple instance AE
f(x_1,x_2,..,x_n) = (h,m_1,m_2,..,m_n)
g(h,m_1,m_2,..,m_n) = x_1,x_2,..,x_n
---
m_i = f_m(x_i,h)
h = f_h(x_1,x_2,..,x_n)
\hat{x}_i = g(h,m_i)
"""
#class MIAE(object):
    
#    def __init__(self):
