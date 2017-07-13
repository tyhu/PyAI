"""
Implementation of VAEs
"""
import sys
import numpy as np
import theano
import theano.tensor as T
from util_theano import *
from compact_cnn import *
import cPickle as pickle

"""
Hierachical VAE
"""

class hVAE(object):
    def __init__(self, struct_h, struct_c, lr=0.001,momentum=0.9):
        x = T.tensor3()
        x_test = T.matrix()
        h, c, ll_test, cost, params = buildHVAE(x, x_test, struct_h, struct_c)
        updates = updatefunc(cost, params.values(), lr=lr, momentum=momentum)
        self.train = theano.function([x], cost, updates=updates)
        self.encode_ = theano.function([x], h)
        self.encode_c_ = theano.function([x], c)
        self.estimate_ = theano.function([x,x_test], ll_test)

        self.tparams = params

    def encode(self, X):
        X = floatX(X)
        return self.encode_(X)

    def encode_c(self, X):
        X = floatX(X)
        return self.encode_c_(X)

    def estimate(self,X,X_test):
        X, X_test = floatX(X), floatX(X_test)
        return self.estimate_(X,X_test)

    def fit_batch(self,X):
        X = floatX(X)
        cost = self.train(X)
        return cost

    def assignParams(self,params):
        for i,tparam in enumerate(self.tparams):
            tparam.set_value(floatX(params[i]))

    def saveParams(self,fn):
        params = []
        for tparam in self.tparams:
            params.append(tparam.get_value())
        pickle.dump(params,open(fn,'wb'))

def hVAE_params(struct_h, struct_c):
    params = {}
    fsize, hsize1,hsize2 = struct_h
    _, csize = struct_c

    rng = np.random.RandomState(12345)
    params['Wh1'] = init_weights_rng((fsize, hsize1), rng)
    params['bh1'] = init_weights_rng((hsize1,), rng)
    params['Wmu'] = init_weights_rng((hsize1, hsize2), rng)
    params['bmu'] = init_weights_rng((hsize2,), rng)
    params['Wsig'] = init_weights_rng((hsize1, hsize2), rng)
    params['bsig'] = init_weights_rng((hsize2,), rng)
    params['Wc'] = init_weights_rng((hsize2+fsize, csize), rng)
    params['bc'] = init_weights_rng((csize,), rng)
    params['Wx'] = init_weights_rng((hsize2+csize, fsize), rng)
    params['bx'] = init_weights_rng((fsize,), rng)
    
    return params

def buildHVAE(x, x_test, struct_h, struct_c):
    batchsize, inum, _ = x.shape
    fsize, hsize1,hsize2 = struct_h
    _, csize = struct_c

    params = hVAE_params(struct_h,struct_c)

    h1 = addFullLayer(x,params['Wh1'],params['bh1'])
    h1p = T.max(h1, axis=1)
    h_mu = addFullLayer(h1p,params['Wmu'],params['bmu'])
    h_sig = addFullLayer(h1p,params['Wsig'],params['bsig'])

    h_sample = sampler_normal((batchsize,hsize2),h_mu,h_sig).dimshuffle(0,'x',1).repeat(inum, axis=1)

    xh = T.concatenate([x,h_sample],axis=2)
    c_dis = addTensorSoftmaxLayer(xh, params['Wc'], params['bc']).reshape((batchsize*inum,csize))
    #c = sampler_cat(c_dis)
    c_enum = T.eye(csize).dimshuffle('x','x',0,1).repeat(batchsize,axis=0).repeat(inum,axis=1)

    h_enum = h_sample.dimshuffle(0,1,'x',2).repeat(csize,axis=2)
    hc_enum = T.concatenate([h_enum,c_enum],axis=3)
    x_enum = addSigmoidFullLayer(hc_enum, params['Wx'], params['bx']).reshape((batchsize*inum, csize, fsize))
    x_dis = T.batched_dot(c_dis,x_enum).reshape((batchsize, inum, fsize))
    #hc = T.concatenate([h_sample,c],axis=2)
    #x_dis = addSigmoidFullLayer(hc, params['Wx'], params['bx'])

    ### cost
    cost = KLD_norm(h_mu, h_sig)
    cost = KLD_bernoulli(c_dis)
    cost = T.mean(cost)
    cost -= bnl_ll(x, x_dis)

    ### p(x_test|x) estimator
    ### P(x|X) ~= \sum_c(p(x,c|h)), where h = q(h|X)
    c_test = T.eye(csize).dimshuffle('x',0,1).repeat(batchsize,axis=0)
    h_test = h_mu.dimshuffle(0,'x',1).repeat(csize, axis=1)
    hc_test = T.concatenate([h_test,c_test],axis=2)
    x_dis_test = addSigmoidFullLayer(hc_test, params['Wx'], params['bx'])
    x_test = x_test.dimshuffle(0,'x',1).repeat(csize, axis=1)
    ll_test = T.mean(T.mean(x_test*T.log(x_dis_test)+(1-x_test)*T.log(1-x_dis_test),axis=2),axis=1)

    return h_mu, c_dis, ll_test, cost, params


### KLD between two gaussian
#def KLD_gauss(mu1, logsig1, mu2, logsig2):
#    batchsize, d = mu1.shape
#    T.log(T.sum(sig1,axis=1)/T.sum(sig2,axis=1))-d+

### KLD with normal distribution
def KLD_norm(mu, logsig):
    return 0.5 * T.sum(1 + logsig - mu**2 - T.exp(logsig), axis=1)
    
### KLD with 0.5,0.5 
def KLD_bernoulli(c_dis):
    _,k = c_dis.shape
    return T.sum(c_dis*T.log(c_dis*k),axis=1)
    
