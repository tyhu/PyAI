import sys
import numpy as np
import theano
import theano.tensor as T
from util_theano import *
from compact_cnn import *
import cPickle as pickle


class TripletEmbNet(object):
    def __init__(self, struct, alpha=0.1, lr=0.001,momentum=0.9):
        x1,x2,x3 = T.matrix(), T.matrix(), T.matrix()
        ltest, cost, params, updates = buildTripletEmb(x1,x2,x3,struct,alpha=alpha)
        updates = updatefunc(cost, params, lr=lr, momentum=momentum) + updates
        self.train = theano.function([x1,x2,x3], cost, updates=updates)
        self.tparams = params
        self.transform_ = theano.function([x1], ltest)
        self.costfunc_ = theano.function([x1,x2,x3],cost)
        
    def costfunc(self,X1,X2,X3):
        X1,X2,X3 = floatX(X1), floatX(X2), floatX(X3)
        return self.costfunc_(X1,X2,X3)

    def fit_batch(self,X1,X2,X3):
        X1,X2,X3 = floatX(X1), floatX(X2), floatX(X3)
        cost = self.train(X1,X2,X3)
        return cost

    def assignParams(self,params):
        for i,tparam in enumerate(self.tparams):
            tparam.set_value(floatX(params[i]))

    def saveParams(self,fn):
        params = []
        for tparam in self.tparams:
            params.append(tparam.get_value())
        pickle.dump(params,open(fn,'wb'))

    def transform(self,X):
        X = floatX(X)
        return self.transform_(X)

"""
x's: input theano tensor
    x1: anchor examples
    x2: positive examples
    x3: negative examples
struct: network structure as a list e.x. [3000,2048,512]
"""
def buildTripletEmb(x1,x2,x3,struct,alpha=0.1):
    fsize, hsize, osize = struct
    rng = np.random.RandomState(12345)
    W1 = init_weights_rng((fsize, hsize), rng)
    b1 = init_weights_rng((hsize,), rng)
    W2 = init_weights_rng((hsize,osize), rng)
    b2 = init_weights_rng((osize,), rng)
    gamma = init_weights((osize,), rng)
    beta = init_weights((osize,), rng)
    params = [W1,b1,W2,b2,gamma,beta]

    h11,h12,h13 = addFullLayer(x1, W1, b1), addFullLayer(x2, W1, b1), addFullLayer(x3, W1, b1)
    h21,h22,h23 = addFullLinearLayer(h11, W2, b2), addFullLinearLayer(h12, W2, b2), addFullLinearLayer(h13, W2, b2)
    bn1, mean, var, bn_updates = addFullBNLayerTrain(h21,gamma,beta)
    bn2, mean, var, _ = addFullBNLayerTrain(h22,gamma,beta,mean=mean,var=var)
    bn3, mean, var, _ = addFullBNLayerTrain(h23,gamma,beta,mean=mean,var=var)

    l1,l2,l3 = l2normalize(bn1), l2normalize(bn2), l2normalize(bn3)

    ### test phase
    bntest = addFullBNLayerTest(h21, gamma, beta, mean, var)
    ltest = l2normalize(bntest)

    ### triplet cost
    #d12 = T.sqrt(1-l1.dot(l2.T))
    #d13 = T.sqrt(1-l1.dot(l3.T))
    d12 = T.sqrt(T.sum((l1-l2)**2,axis=1))
    d13 = T.sqrt(T.sum((l1-l3)**2,axis=1))
    cost = T.mean(T.maximum(d12-d13+alpha,0))

    return ltest, cost, params, bn_updates

