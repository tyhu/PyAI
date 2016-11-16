"""
Autoencoder
Ting-Yao Hu, 2016.10
"""
import numpy as np
import theano
import theano.tensor as T
from util_theano import *

'''
ld2 -- l2 norm weight
lr --
momentum --
tied -- if True, encoding and decoding matrix are the same
'''
class AutoEncoder(object):
    
    def __init__(self, shape, activation='sigmoid',lr=0.0001,momentum=0.9,ld2=0.001,tied=True):
        featsize, hsize = shape
        x = T.matrix()
        xtar = T.matrix()
        rng = np.random.RandomState(1245)
        We = init_weights_rng((featsize,hsize),rng)
        be = init_weights_rng((hsize,),rng)
        Wd = init_weights_rng((hsize,featsize),rng)
        bd = init_weights_rng((featsize,),rng)
        h, cost, params = buildAE(x,xtar,We,be,Wd,bd,activation,ld2=ld2,tied=tied)

        updates = updatefunc(cost, params, lr=lr, momentum=momentum)
        self.train = theano.function([x,xtar], cost, updates=updates)
        self.transform_ = theano.function([x], h)

    """
    X -- training data in shape (datanum, featnum) (numpy array)
    """
    def fit(self,X,batchsize=50):
        X = floatX(X)
        dnum = X.shape[0]
        for i in range(500):
            print "iteration %d" % (i + 1)
            totalcost = 0
            for start in range(0, dnum, batchsize):
                x_batch = X[start:start + batchsize]
                cost = self.train(x_batch,x_batch)
                totalcost+=cost
            print "cost: ",totalcost

    def fit_denoise(self,X,batchsize=50,p=0.5):
        X = floatX(X)
        dnum = X.shape[0]

        for i in range(1000):
            #print "iteration %d" % (i + 1)
            totalcost = 0
            for start in range(0, dnum, batchsize):
                x_batch = X[start:start + batchsize]
                mask = np.random.binomial(1,p,size=x_batch.shape)
                x_batch_corrupt = floatX(x_batch*mask)
                cost = self.train(x_batch_corrupt,x_batch)
                totalcost+=cost
            #print "cost: ",totalcost

    def transform(self,X):
        X = floatX(X)
        return self.transform_(X)

def buildAE(x,xtar,We,be,Wd,bd,activation='sigmoid',ld2=0.001, tied=True):
    h = T.dot(x,We)+be

    if activation is 'sigmoid':
        h = T.nnet.sigmoid(h)
        if tied: xout = T.nnet.sigmoid(T.dot(h,We.T)+bd)
        else: xout = T.nnet.sigmoid(T.dot(h,Wd)+bd)
    elif activation is 'relu':
        h = T.maximum(h,0)
        if tied: xout = T.maximum(T.dot(h,We.T)+bd,0)
        else: xout = T.maximum(T.dot(h,Wd)+bd,0)
    elif activation is 'tanh':
        h = T.tanh(h)
        if tied: xout = T.tanh(T.dot(h,We.T)+bd)
        else: xout = T.tanh(T.dot(h,Wd)+bd)
    else:
        raise NotImplementedError()

    if tied: params = [We,be,bd]
    else: params = [We,be,Wd,bd]

    cost = MSE(xtar,xout)+l2norm([We])*ld2
    return h, cost, params

'''
domain adaptation autoencoder
shape -- (featsize, hidden layer size, transfer layer size)
'''
class DAAE(object):
    def __init__(self, shape, activation='sigmoid',lr=0.0001,momentum=0.9,ld2=0.001,tied=True):
        fsize, hsize, tsize = shape
        rng = np.random.RandomState(1245)
        x = T.matrix()
        xtar = T.matrix()
        W = init_weights_rng((fsize,hsize),rng)
        be = init_weights_rng((hsize,),rng)
        bd = init_weights_rng((fsize,),rng)
        Wt = init_weights_rng((fsize,tsize),rng)
        bte = init_weights_rng((tsize,),rng)
        btd = init_weights_rng((fsize,),rng)
        h, cost_s, params_s, ht, cost_t, params_t = buildDAAE(x,xtar,W,be,bd,Wt,bte,btd,activation=activation,ld2=ld2)
        updates = updatefunc(cost_s, params_s, lr=lr, momentum=momentum)
        updates_t = updatefunc(cost_t, params_t, lr=lr, momentum=momentum)
        self.train_s = theano.function([x,xtar], cost_s, updates=updates)
        self.train_t = theano.function([x,xtar], cost_t, updates=updates_t)
        self.transform_ = theano.function([x], h)
        
    def fit_denoise(self,X_s,X_t,batchsize=50,p=0.5):
        X_s,X_t = floatX(X_s), floatX(X_t)
        snum = X_s.shape[0]
        tnum = X_t.shape[0]

        for i in range(1000):
            #print "iteration %d" % (i + 1)
            totalcost = 0
            for start in range(0, snum, batchsize):
                x_batch = X_s[start:start + batchsize]
                mask = np.random.binomial(1,p,size=x_batch.shape)
                x_batch_corrupt = floatX(x_batch*mask)
                cost = self.train_s(x_batch_corrupt,x_batch)
                totalcost+=cost
            for start in range(0, tnum, batchsize):
                x_batch = X_t[start:start + batchsize]
                mask = np.random.binomial(1,p,size=x_batch.shape)
                x_batch_corrupt = floatX(x_batch*mask)
                cost = self.train_t(x_batch_corrupt,x_batch)
                totalcost+=cost
            #print "cost: ",totalcost

    def transform(self,X):
        X = floatX(X)
        return self.transform_(X)

def buildDAAE(x,xtar,W,be,bd,Wt,bte,btd,activation='sigmoid',ld2=0.001):
    if activation is 'sigmoid': activ_func = T.nnet.sigmoid
    elif activation is 'relu': activ_func = ReLU
    elif activation is 'tanh': activ_func = T.tanh
    else: raise NotImplementedError()
    ### source domain autoencoder
    h = activ_func(T.dot(x,W)+be)
    xout_s = activ_func(T.dot(h,W.T)+bd)

    ### target domain autoencoder
    ht = activ_func(T.dot(x,W)+be)
    htd = activ_func(T.dot(x,Wt)+bte)
    xout_t = activ_func(T.dot(ht,W.T)+bd+T.dot(htd,Wt.T)+btd)

    params_s = [W,be,bd]
    params_t = [W,be,bd,Wt,bte,btd]
    cost_s = MSE(xtar,xout_s)+l2norm(params_s)*ld2
    cost_t = MSE(xtar,xout_t)+l2norm(params_t)*ld2
    
    return h, cost_s, params_s, ht, cost_t, params_t


class JFAE(object):
    def __init__(self, shape, activation='sigmoid',lr=0.0001,momentum=0.9,ld2=0.001,tied=True):
        fsize, hsize, dsize = shape
        rng = np.random.RandomState(1245)
        x = T.matrix()
        xtar = T.matrix()
        y = T.ivector('y')
        W = init_weights_rng((fsize,hsize),rng)
        b = init_weights_rng((hsize,),rng)
        bo = init_weights_rng((fsize,),rng)
        self.Wd = init_weights_zeros((hsize,dsize))
        bd = init_weights_rng((dsize,),rng)
        h, cost, params = buildJFAE(x,xtar,y,W,b,bo,self.Wd,bd,activation=activation,ld2=ld2)

        updates = updatefunc(cost, params, lr=lr, momentum=momentum)
        self.train = theano.function([x,xtar,y], cost, updates=updates)
        self.transform_ = theano.function([x], h)

    def fit_denoise(self,X_s,X_t,batchsize=50,p=0.5):
        X_s,X_t = floatX(X_s), floatX(X_t)
        snum = X_s.shape[0]
        tnum = X_t.shape[0]
        y_s = [0]*snum
        y_t = [1]*tnum
        X = np.concatenate((X_s,X_t),axis=0)
        y = np.array(y_s+y_t).astype('int32')

        per = np.random.permutation(X.shape[0])
        X,y = X[per],y[per]

        for i in range(1000):
            print "iteration %d" % (i + 1)
            totalcost = 0
            for start in range(0, snum+tnum, batchsize):
                x_batch = X[start:start + batchsize]
                mask = np.random.binomial(1,p,size=x_batch.shape)
                x_batch_corrupt = floatX(x_batch*mask)
                y_batch = y[start:start + batchsize]
                cost = self.train(x_batch_corrupt,x_batch,y_batch)
                totalcost+=cost
            print "cost: ",totalcost
            print np.where(self.Wd.get_value()<0.00001)

    def transform(self,X):
        X = floatX(X)
        Wd = self.Wd.get_value()
        print np.where(Wd==0)
        mask = np.where(np.abs(Wd).sum(axis=1)==0)[0]
        out = self.transform_(X)
        out = out[:,mask]
        return out


"""
Joint Factor AutoEncoder
"""
def buildJFAE(x,xtar,y,W,b,bo,Wd,bd,activation='sigmoid',ld2=0.001):
    if activation is 'sigmoid': activ_func = T.nnet.sigmoid
    elif activation is 'relu': activ_func = ReLU
    elif activation is 'tanh': activ_func = T.tanh
    else: raise NotImplementedError()

    h = activ_func(T.dot(x,W)+b)
    dout = T.nnet.softmax(T.dot(h,Wd)+bd)
    xout = activ_func(T.dot(h,W.T)+bo)

    params = [W,b,bo,Wd,bd]
    ld1 = 1
    ld3 = 1
    cost = MSE(xtar,xout)+l2norm(params)*ld2+neglogL(y,dout)*ld3+l1norm([Wd])*ld1
    #cost = MSE(xtar,xout)+l2norm(params)*ld2

    return h, cost, params

class DualAE(object):
    def __init__(self, shape, activation='sigmoid',lr=0.0001,momentum=0.9,ld2=0.001,tied=True):
        featsize, hsize = shape
        x = T.matrix()
        xtar = T.matrix()
        rng = np.random.RandomState(1245)
        Wst = init_weights_rng((featsize,hsize),rng)
        bst = init_weights_rng((hsize,),rng)
        Wts = init_weights_rng((hsize,featsize),rng)
        bts = init_weights_rng((featsize,),rng)
        h_st, cost_st, params_st = buildDualAE(x,xtar,Wst,bst,Wts,bts,activation,ld2=ld2)
        h_ts, cost_ts, params_ts = buildDualAE(x,xtar,Wts,bts,Wst,bst,activation,ld2=ld2)

        updates_st = updatefunc(cost_st, params_st, lr=lr, momentum=momentum)
        updates_ts = updatefunc(cost_st, params_st, lr=lr, momentum=momentum)
        self.train_st = theano.function([x,xtar], cost_st, updates=updates_st)
        self.train_ts = theano.function([x,xtar], cost_ts, updates=updates_ts)
        self.transform_ = theano.function([x], h_st)

    def fit(self,Xs,Xt,batchsize=50):
        Xs,Xt = floatX(Xs),floatX(Xt)
        dnum = X.shape[0]
        for i in range(1000):
            print "iteration %d" % (i + 1)
            totalcost = 0
            for start in range(0, dnum, batchsize):
                x_batch = X[start:start + batchsize]
                cost = self.train(x_batch,x_batch)
                totalcost+=cost
            print "cost: ",totalcost


def buildDualAE(x,xtar,Wi,Wo,bi,bo,activation='sigmoid',ld2=0.001):
    if activation is 'sigmoid': activ_func = T.nnet.sigmoid
    elif activation is 'relu': activ_func = ReLU
    elif activation is 'tanh': activ_func = T.tanh
    else: raise NotImplementedError()
    
    h = activ_func(T.dot(x,Wi)+bi)
    xout = activ_func(T.dot(h,Wo.T)+bo)

    params = [Wi,bi,Wo,bo]
    cost = MSE(xtar,xout)+l2norm([Wi,Wo])*ld2

    return h,cost,params

