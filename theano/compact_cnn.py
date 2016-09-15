import numpy as np

import theano
import theano.tensor as T
from theano.tensor.nnet import conv2d
from theano.tensor.signal.pool import pool_2d

def floatX(x):
    return np.asarray(x, dtype=theano.config.floatX)

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

"""
add one convolution layer with relu, max pooling
x -- (batchsize,fmaps num,img_height,img_width)
W -- (out maps num, in maps num, filter_height, filter_width)
b -- (out maps num)
"""
def addConvPoolLayer(x,W,b,border_mode='valid',psize=(2,2)):
    #c = T.maximum(0, conv2d(x, W) + b.dimshuffle('x', 0, 'x', 'x'))
    #p = pool_2d(c, psize,ignore_border=True)
    c = addConvLayer(x,W,b,border_mode=border_mode)
    p = addPoolLayer(c,psize=psize)
    return p

def addConvLayer(x,W,b,border_mode='valid'):
    c = T.maximum(0, conv2d(x, W, border_mode=border_mode) + b.dimshuffle('x', 0, 'x', 'x'))
    return c

def addPoolLayer(x,psize=(2,2)):
    p = pool_2d(x, psize, ignore_border=True)
    return p

def addFullLayer(x,W,b):
    out = T.maximum(0, T.dot(x, W) + b)
    return out

def addSoftmaxLayer(x,W,b):
    p_y_given_x = T.nnet.softmax(T.dot(x, W) + b)
    return p_y_given_x

### negative log likelihood
def negll(p_y_given_x, t):
    return T.mean(T.nnet.categorical_crossentropy(p_y_given_x, t))

def updatefunc(cost, params, lr, momentum):
    grads = theano.grad(cost, params)
    updates = []
    for p, g in zip(params, grads):
        mparam_i = theano.shared(np.zeros(p.get_value().shape, dtype=theano.config.floatX))
        v = momentum * mparam_i - lr * g
        updates.append((mparam_i, v))
        updates.append((p, p + v))
    return updates
