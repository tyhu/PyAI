import numpy as np

import theano
import theano.tensor as T
from theano.tensor.nnet import conv2d
from theano.tensor.signal.pool import pool_2d

def ReLU(x):
    return T.maximum(0,x)

def floatX(x):
    return np.asarray(x, dtype=theano.config.floatX)

def init_weights_zeros(shape):
    return theano.shared(floatX(np.zeros(shape)))

def init_weights(shape, rng):
    return theano.shared(floatX(np.random.randn(*shape) * 0.1))

def init_weights_rng(shape,rng):
    if len(shape)==1: return theano.shared(floatX(np.zeros(shape)))

    fan_in = np.prod(shape[1:])
    fan_out = (shape[0] * np.prod(shape[2:]))/4
    W_bound = np.sqrt(6./(fan_in + fan_out))
    W = theano.shared(
        np.asarray(
            rng.uniform(low=-W_bound, high=W_bound, size=shape),
            dtype=theano.config.floatX
        ),
        borrow=True
    )
    return W

def l2norm(params):
    l2 = (params[0]**2).sum()
    for param in params[1:]:
        l2 = l2+(param**2).sum()
    return l2

def l1norm(params):
    l1 = abs(params[0]).sum()
    for param in params[1:]:
        l1 = l1+abs(param).sum()
    return l1

def MSE(ytest,ypred):
    dnum = ytest.shape[0]
    return T.sum(T.pow(ytest-ypred,2))/dnum

### negative log likelihood
### y -- ouput
### t -- target
def neglogL(t,y):
    return -T.mean(T.log(y)[T.arange(t.shape[0]), t])

def updatefunc(cost, params, lr, momentum):
    grads = theano.grad(cost, params)
    updates = []
    for p, g in zip(params, grads):
        mparam_i = theano.shared(np.zeros(p.get_value().shape, dtype=theano.config.floatX))
        v = momentum * mparam_i - lr * g
        updates.append((mparam_i, v))
        updates.append((p, p + v))
    return updates
