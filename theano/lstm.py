import sys
import numpy as np
import theano
import theano.tensor as T
from util_theano import *
from collections import OrderedDict


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
def lstm_layer(X,mask,shape,tparams,prefix='lstm'):
    #samplesize, tsize, idim = X.shape
    tsize, samplesize, idim = X.shape
    idim, hdim = shape

    ### add new tparams
    U,b,W = tparams[pp(prefix,'U')], tparams[pp(prefix,'b')], tparams[pp(prefix,'W')]

    init_state = T.alloc(0., samplesize, hdim)
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
    lstm = LSTMClassifier((1000,512),11)



if __name__=='__main__':
    test()
