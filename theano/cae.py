import sys
sys.path.append('3rdparty/')
import numpy as np
import theano
import theano.tensor as T
from util_theano import *
from compact_cnn import *
import cPickle as pickle


'''
convolutional autoencoder
X: theano shared variable, (datanum, feat_map_num, x, y)
'''

class CAE(object):

    def __init__(self,lr=0.001,momentum=0.9,ld2=0.01):
        x = T.tensor4()
        o, cost, params = buildCAE2(x)
        #o, cost, params = buildCAE(x)
        updates = updatefunc(cost, params, lr=lr, momentum=momentum)
        self.train = theano.function([x], cost, updates=updates)
        self.cost = theano.function([x],cost)
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

    def fit(self, X, batchsize=32, iternum=1000):
        X = floatX(X)
        mincost = None
        accu = 0
        for i in range(iternum):
            totalcost = 0
            for start in range(0, len(X), batchsize):
                x_batch = X[start:start + batchsize]
                cost = self.train(x_batch)
                totalcost+=cost
            print 'totalcost: ',totalcost
            ### early stop
            if mincost is None or mincost>totalcost:
                mincost = totalcost
                accu = 0
            else: accu+=1
            if accu>10:
                break
                print 'early stop, cost:', totalcost

    def valid(self,X,batchsize=32):
        X = floatX(X)
        validcost = 0.0
        for start in range(0, len(X), batchsize):
            x_batch = X[start:start + batchsize]
            cost = self.cost(x_batch)
            validcost+=cost
        return validcost

    def transform(self,X):
        X = floatX(X)
        return self.transform_(X)

class EmCAE(CAE):
    def __init__(self,lr=0.01,ld=0.1,momentum=0.9,ld2=0.01):
        x1,x2 = T.tensor4(), T.tensor4()
        o, rcost, ecost, params, eparams = buildEmCAE(x1,x2)
        updates_r = updatefunc(rcost, params, lr=lr, momentum=momentum)
        updates_e = updatefunc(ecost, eparams, lr=lr*ld, momentum=momentum)
        self.rtrain = theano.function([x1], rcost, updates=updates_r)
        self.etrain = theano.function([x1,x2], ecost, updates=updates_e)
        self.rcost = theano.function([x1], rcost)
        self.ecost = theano.function([x1,x2], ecost)
        self.transform_ = theano.function([x1], o)
        self.tparams = params
    
    ### X1, X2 are patch pairs found from surveillance
    def fit_joint(self, X, X1, X2, batchsize=32, iternum=10):
        X,X1,X2 = floatX(X), floatX(X1), floatX(X2)
        mincost = None
        totalcost_r, totalcost_e = 0,0
        for i in range(iternum):
            totalcost_r, totalcost_e = 0,0
            ### reconstruction
            for start in range(0, len(X), batchsize):
                x_batch = X[start:start + batchsize]
                cost = self.rtrain(x_batch)
                totalcost_r+=cost
            #print 'totalcost: ',totalcost
            
            ### embeding
            #for start in range(0, len(X1), batchsize):
                estart = start%len(X1)
                x1_batch = X1[estart:estart + batchsize]
                x2_batch = X2[estart:estart + batchsize]
                ecost = self.etrain(x1_batch,x2_batch)
                #ecost = self.ecost(x1_batch,x2_batch)
                totalcost_e+=ecost
            
            print 'reconstruction cost: ',totalcost_r
            print 'embeding cost: ',totalcost_e
        

def unpoolLayer(X):
    Xup = X.repeat(2,axis=2).repeat(2,axis=3)
    return Xup

def unflattenLayer(X, shape):
    return X.reshape(shape)

def buildCAE2(x):
    filternum = 20
    #filternum = 40
    rng = np.random.RandomState(12345)
    w_c1 = init_weights_rng((filternum, 3, 5, 5),rng)
    b_c1 = init_weights_rng((filternum,),rng)
    #w_dc1 = w_c1.dimshuffle((1,0,2,3))
    w_dc1 = init_weights_rng((3, filternum, 5, 5),rng)
    b_dc1 = init_weights_rng((3,),rng)
    w_h = init_weights_rng((filternum * 16 * 16, 200),rng)
    b_h = init_weights_rng((200,),rng)
    b_dh = init_weights_rng((filternum*16*16,),rng)

    c1 = addSigmoidConvLayer(x,w_c1,b_c1,border_mode=2)
    p1 = addPoolLayer(c1,psize=(2,2))
    flat = p1.flatten(2)
    o = addSigmoidFullLayer(flat,w_h,b_h)   ###encoder
    h = addSigmoidFullLayer(o,w_h.T,b_dh)
    dflat = unflattenLayer(h, p1.shape)
    dp1 = unpoolLayer(dflat)
    dc1 = addSigmoidConvLayer(dp1,w_dc1,b_dc1,border_mode=2)

    ld2 = 0.001
    datanum = x.shape[0]
    #cost = MSE(x,dc1)/datanum+ld2*l2norm([w_c1,w_h])
    cost = MSE(x,dc1)/datanum+ld2*l2norm([w_c1,w_dc1,w_h])
    #params = [w_c1,b_c1,b_dc1,w_h,b_h,b_dh]
    params = [w_c1,b_c1,w_dc1,b_dc1,w_h,b_h,b_dh]
    return o, cost, params

def buildEmCAE(x1,x2):
    filternum = 20
    #filternum = 40
    rng = np.random.RandomState(12345)
    w_c1 = init_weights_rng((filternum, 3, 5, 5),rng)
    b_c1 = init_weights_rng((filternum,),rng)
    #w_dc1 = w_c1.dimshuffle((1,0,2,3))
    w_dc1 = init_weights_rng((3, filternum, 5, 5),rng)
    b_dc1 = init_weights_rng((3,),rng)
    w_h = init_weights_rng((filternum * 16 * 16, 200),rng)
    b_h = init_weights_rng((200,),rng)
    b_dh = init_weights_rng((filternum*16*16,),rng)

    c1 = addSigmoidConvLayer(x1,w_c1,b_c1,border_mode=2)
    p1 = addPoolLayer(c1,psize=(2,2))
    flat = p1.flatten(2)
    o1 = addSigmoidFullLayer(flat,w_h,b_h)   ###encoder
    h = addSigmoidFullLayer(o1,w_h.T,b_dh)
    dflat = unflattenLayer(h, p1.shape)
    dp1 = unpoolLayer(dflat)
    dc1_1 = addSigmoidConvLayer(dp1,w_dc1,b_dc1,border_mode=2)

    ### second network
    c1 = addSigmoidConvLayer(x2,w_c1,b_c1,border_mode=2)
    p1 = addPoolLayer(c1,psize=(2,2))
    flat = p1.flatten(2)
    o2 = addSigmoidFullLayer(flat,w_h,b_h)   ###encoder
    #h = addSigmoidFullLayer(o2,w_h.T,b_dh)
    #dflat = unflattenLayer(h, p1.shape)
    #dp1 = unpoolLayer(dflat)
    #dc1_2 = addSigmoidConvLayer(dp1,w_dc1,b_dc1,border_mode=2)

    ld2 = 0.001
    datanum1,datanum2 = x1.shape[0], x2.shape[0]
    recon_cost = MSE(x1,dc1_1)/datanum1
    recon_cost+=ld2*l2norm([w_c1,w_dc1,w_h])
    em_cost = MSE(o1,o2)/datanum2
    params = [w_c1,b_c1,w_dc1,b_dc1,w_h,b_h,b_dh]
    eparams = [w_c1,b_c1,w_h,b_h]
    return o1, recon_cost, em_cost, params, eparams

def buildCAE(x):
    rng = np.random.RandomState(12345)
    w_c1 = init_weights_rng((30, 3, 5, 5),rng)
    b_c1 = init_weights_rng((30,),rng)
    w_dc1 = init_weights_rng((3, 30, 5, 5),rng)
    b_dc1 = init_weights_rng((3,),rng)
    w_c2 = init_weights_rng((30, 30, 5, 5),rng)
    b_c2 = init_weights_rng((30,),rng)
    w_dc2 = init_weights_rng((30, 30, 5, 5),rng)
    b_dc2 = init_weights_rng((30,),rng)
    w_c3 = init_weights_rng((20, 30, 5, 5),rng)
    b_c3 = init_weights_rng((20,),rng)
    w_dc3 = init_weights_rng((30, 20, 5, 5),rng)
    b_dc3 = init_weights_rng((30,),rng)
    w_h = init_weights_rng((20 * 4 * 4, 200),rng)
    b_h = init_weights_rng((200,),rng)
    b_dh = init_weights_rng((20*4*4,),rng)

    c1 = addSigmoidConvLayer(x,w_c1,b_c1,border_mode=2)
    ### TODO batch normalization
    p1 = addPoolLayer(c1,psize=(2,2))
    c2 = addSigmoidConvLayer(p1,w_c2,b_c2,border_mode=2)
    p2 = addPoolLayer(c2,psize=(2,2))
    c3 = addSigmoidConvLayer(p2,w_c3,b_c3,border_mode=2)
    p3 = addPoolLayer(c3,psize=(2,2))
    flat = p3.flatten(2)
    o = addSigmoidFullLayer(flat,w_h,b_h)   ###encoder
    h = addSigmoidFullLayer(o,w_h.T,b_dh)
    dflat = unflattenLayer(h, p3.shape)
    dp3 = unpoolLayer(dflat)
    dc3 = addSigmoidConvLayer(dp3,w_dc3,b_dc3,border_mode=2)
    dp2 = unpoolLayer(dc3)
    dc2 = addSigmoidConvLayer(dp2,w_dc2,b_dc2,border_mode=2)
    dp1 = unpoolLayer(dc2)
    dc1 = addSigmoidConvLayer(dp1,w_dc1,b_dc1,border_mode=2)

    ld2 = 0.001
    cost = MSE(x,dc1)+ld2*l2norm([w_c1,w_dc1,w_c2,w_dc2,w_c3,w_dc3,w_h])
    params = [w_c1,b_c1,b_dc1,w_c2,b_c2,b_dc2,w_c3,b_c3,b_dc3,w_h,b_h,b_dh]
    return o, cost, params

def test():
    print 'test'
    Tx = T.tensor4()
    flat = Tx.flatten(2)
    unflat = flat.reshape(Tx.shape)
    f = theano.function([Tx],unflat)
    x = np.random.rand(2,1,3,2)
    y = f(x)
    print x
    print y

if __name__=='__main__':
    test()
