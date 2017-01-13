import sys
import numpy as np
import theano
import theano.tensor as T
from util_theano import *
from collections import OrderedDict


def rnn_params(params,shape,prefix='rnn'):
    idim, hdim = shape
    W = norm_weight(idim,hdim)
    U = ortho_weight(hdim)
    params[pp(prefix,'W')] = W
    params[pp(prefix,'U')] = U
    params[pp(prefix,'b')] = np.zeros((hdim,)).astype('float32')

def rnn_layer(X,shape,tparams,init_state=None,prefix='rnn'):
    tsize, samplesize, idim = X.shape
    idim, hdim = shape
    ### add new tparams
    U,b,W = tparams[pp(prefix,'U')], tparams[pp(prefix,'b')], tparams[pp(prefix,'W')]

    def _step(xtran_,htm1_):
        ht_ = T.nnet.sigmoid(xtran_+T.dot(htm1_,U))
        #ht_ = ReLU(xtran_+T.dot(htm1_,U))
        return ht_

    if init_state is None:
        init_state = T.alloc(0., samplesize, hdim)
    Xtrans = T.dot(X,W)+b
    rval, updates = theano.scan(
        _step,
        sequences=Xtrans,
        outputs_info = init_state,
        n_steps=tsize,
    )
    return rval

def decoder_params(params,shape,prefix='decoder'):
    idim,hdim = shape
    Wde = norm_weight(hdim,idim)
    bd = np.zeros((idim,)).astype('float32')
    params[pp(prefix,'Wde')] = Wde
    params[pp(prefix,'bd')] = bd

    
'''
X -- input tensor
shape -- [input_dim, hidden(representation)_dim]
'''
def build_rnnae(X,shape):
    idim, hdim = shape
    nsteps, nsample, _ = X.shape
    
    params = {}
    #rnn_params(params, shape, prefix='encoder_rnn')
    #rnn_params(params, shape, prefix='decoder_rnn')
    lstm_params(params, shape, prefix='encoder_lstm')
    lstm_params(params, shape, prefix='decoder_lstm')
    decoder_params(params, shape, prefix='decoder')
    tparams = init_tparams(params)

    #h = rnn_layer(X, shape, tparams, prefix='encoder_rnn')
    h = lstm_layer(X, X, shape, tparams, prefix='encoder_lstm')
    rep = h[-1]

    Wde = tparams['decoder_Wde']
    bd = tparams['decoder_bd']
    o_first = T.dot(rep,Wde)+bd
    #o_first = T.nnet.sigmoid(o_first)
    Xr = X[::-1]
    #hd = rnn_layer(Xr[:-1], shape, tparams, prefix='decoder_rnn',init_state=rep)
    hd = lstm_layer(Xr[:-1], Xr[:-1], shape, tparams, prefix='decoder_lstm',init_state=rep)
    o = T.dot(hd,Wde)+bd
    #o = T.nnet.sigmoid(o)

    #cost = MSE(o, Xr[1:])
    cost = T.sum(T.pow(o-Xr[1:],2))+T.sum(T.pow(o_first-Xr[0],2))
    tlr = T.scalar(name='lr')
    grads = T.grad(cost, wrt=list(tparams.values()))
    f_cost1, f_train = adadelta(tlr, tparams, grads,
        [X], cost)
    #f_cost = theano.function([X], cost, name='f_cost')
    f_extract = theano.function([X], rep)

    return f_cost1, f_train, f_extract

def build_rnnclf(shape, labsize, ld=0.01):
    X = T.tensor3('X',dtype='float32')
    y = T.vector('y', dtype='int64')

    tsize, nsample, idim = X.shape
    idim, hdim = shape

    params = {}
    rnn_params(params, shape, prefix='rnn')
    params['Wc'] = norm_weight(hdim,labsize)
    params['bc'] = floatX(np.zeros((labsize,)))
    tparams = init_tparams(params)

    h = rnn_layer(X, shape, tparams, prefix='rnn')
    rep = h[-1]
    pred = T.nnet.softmax(T.dot(rep,tparams['Wc'])+tparams['bc'])

    off = 1e-8
    cost = -T.log(pred[T.arange(nsample), y] + off).mean()
    cost+=l2norm([tparams['Wc']])*ld
    for k in tparams.keys():
        if 'W' in k or 'U' in k: cost+=l2norm([tparams[k]])*ld

    tlr = T.scalar(name='lr')
    grads = T.grad(cost, wrt=list(tparams.values()))
    f_cost1, f_train = adadelta(tlr, tparams, grads,
        [X, y], cost)
    f_extract = theano.function([X], rep)
    return f_cost1, f_train, f_extract

'''
params -- parameter dict
'''
def lstm_params(params,shape,prefix='lstm'):
    idim, hdim = shape
    W = np.concatenate([norm_weight(idim,hdim),
        norm_weight(idim,hdim),
        norm_weight(idim,hdim),
        norm_weight(idim,hdim)], axis=1)
    params[pp(prefix,'W')] = W
    U = np.concatenate([ortho_weight(hdim),
        ortho_weight(hdim),
        ortho_weight(hdim),
        ortho_weight(hdim)], axis=1)
    params[pp(prefix,'U')] = U
    params[pp(prefix,'b')] = np.zeros((4 * hdim,)).astype('float32')
    
    return params

'''
X (T.matrix) -- input sequence (nsample ,timesteps, featdim)
shape (list) -- shape[0]: featdim, shape[1]: hidden dim
'''
def lstm_layer(X,mask,shape,tparams,init_state=None,prefix='lstm'):
    #samplesize, tsize, idim = X.shape
    tsize, samplesize, idim = X.shape
    idim, hdim = shape

    ### add new tparams
    U,b,W = tparams[pp(prefix,'U')], tparams[pp(prefix,'b')], tparams[pp(prefix,'W')]

    if init_state is None: init_state = T.alloc(0., samplesize, hdim)
    init_memory = T.alloc(0., samplesize, hdim)

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n*dim:(n+1)*dim]
        elif _x.ndim == 2:
            return _x[:, n*dim:(n+1)*dim]
        return _x[n*dim:(n+1)*dim]

    def _step(m_, x_, h_, c_):
        preact = T.dot(h_, U)
        preact += x_

        i = T.nnet.sigmoid(_slice(preact, 0, hdim))
        f = T.nnet.sigmoid(_slice(preact, 1, hdim))
        o = T.nnet.sigmoid(_slice(preact, 2, hdim))
        c = T.tanh(_slice(preact, 0, hdim))
        
        c = f * c_ + i * c
        #c = m_[:, None] * c + (1. - m_)[:, None] * c_

        h = o * T.tanh(c)
        #h = m_[:, None] * h + (1. - m_)[:, None] * h_

        return h, c

    Xtrans = T.dot(X,W)+b
    rval, updates = theano.scan(
        _step,
        sequences=[mask, Xtrans],
        outputs_info = [init_state,init_memory],
        n_steps=tsize,
    )
    return rval[0]


class LSTMClassifier(object):

    def __init__(self,shape,labsize,ld2=0.01, optimizer=adadelta):
        idim,hdim = shape
        params = {}
        params = lstm_params(params,shape)
        params['Wc'] = norm_weight(hdim,labsize)
        params['bc'] = floatX(np.zeros((labsize,)))
        tparams = init_tparams(params)

        X = T.tensor3('X',dtype='float32')
        nsteps, nsample, _ = X.shape
        mask = T.matrix('mask', dtype='float32')
        y = T.vector('y', dtype='int64')
        h = lstm_layer(X,mask,shape,tparams)
        h = h.mean(axis=0)
        pred = T.nnet.softmax(T.dot(h,tparams['Wc'])+tparams['bc'])

        off = 1e-8
        cost = -T.log(pred[T.arange(nsample), y] + off).mean()
        cost+=l2norm([tparams['Wc']])
        
        ###
        self.f_pred_prob = theano.function([X, mask], pred, name='f_pred_prob')
        self.f_pred = theano.function([X, mask], pred.argmax(axis=1), name='f_pred')
        tlr = T.scalar(name='lr')
        grads = T.grad(cost, wrt=list(tparams.values()))
        self.f_grad_shared, self.f_update = optimizer(tlr, tparams, grads,
            [X, mask, y], cost)
        self.f_cost = theano.function([X, mask, y], cost, name='f_cost')

    def fit_batch(self,X,y):
        batchsize ,nsteps, idim = X.shape
        mask = np.ones((batchsize,nsteps))

    def fit(self,X,y,batchsize=2):
        nsteps, nsample, idim = X.shape
        mask = floatX(np.ones((nsteps,batchsize)))
        for i in range(500):
            print "iteration %d" % (i+1)
            totalcost = 0
            for start in range(0, nsample, batchsize):
                x_batch = X[:,start:start + batchsize,:]
                y_batch = y[start:start + batchsize]
                cost = self.f_grad_shared(x_batch, mask, y_batch)
                self.f_update(0.01)
                totalcost+=cost
            print "cost: ",totalcost

    def predict(self,X):
        nsteps, nsamples, idim = X.shape
        ypred = []
        mask = floatX(np.ones((1,nsteps)))
        for i in range(nsamples):
            x = X[:,i:i+1,:]
            y = self.f_pred(x,mask)
            ypred.append(y[0])
        return ypred
            

def test():
    #lstm = LSTMClassifier((1000,512),11)

    X = T.tensor3('X',dtype='float32')
    nsteps, nsample, _ = X.shape
    shape = [26,10]
    f_cost,f_train = build_rnnae(X,shape)
    


if __name__=='__main__':
    test()
