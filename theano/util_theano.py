import numpy as np

import theano
import theano.tensor as T
from theano.tensor.nnet import conv2d
from theano.tensor.signal.pool import pool_2d
from collections import OrderedDict

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


# make prefix-appended name
def pp(pp, name):
    return '%s_%s'%(pp, name)

def init_tparams(params):
    tparams = OrderedDict()
    for kk, pp in params.items():
        tparams[kk] = theano.shared(params[kk], name=kk)
    return tparams

def getNumpyRNG(seed=None):
    if seed is None: seed = 1234
    rng_numpy = np.random.RandomState(seed)
    return rng_numpy
    

def ortho_weight(ndim, seed=None):
    """
    Random orthogonal weights, we take
    the right matrix in the SVD.

    Remember in SVD, u has the same # rows as W
    and v has the same # of cols as W. So we
    are ensuring that the rows are 
    orthogonal. 
    """
    rng_numpy = getNumpyRNG()
    W = rng_numpy.randn(ndim, ndim)
    u, _, _ = np.linalg.svd(W)
    return u.astype('float32')

def norm_weight(nin,nout=None, scale=0.01, ortho=True):
    """
    Random weights drawn from a Gaussian
    """
    rng_numpy = getNumpyRNG()
    if nout == None:
        nout = nin
    if nout == nin and ortho:
        W = ortho_weight(nin)
    else:
        W = scale * rng_numpy.randn(nin, nout)
    return W.astype('float32')

# optimizers
# name(hyperp, tparams, grads, inputs (list), cost) = f_grad_shared, f_update
def adadelta(lr, tparams, grads, inp, cost):
    zipped_grads = [theano.shared(p.get_value() * floatX(0.), name='%s_grad'%k) for k, p in tparams.iteritems()]
    running_up2 = [theano.shared(p.get_value() * floatX(0.), name='%s_rup2'%k) for k, p in tparams.iteritems()]
    running_grads2 = [theano.shared(p.get_value() * floatX(0.), name='%s_rgrad2'%k) for k, p in tparams.iteritems()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    #rg2up = [(rg2, floatX(0.95) * rg2 + floatX(0.05) * (g ** 2)) for rg2, g in zip(running_grads2, grads)]
    rg2up = [(rg2, rg2*floatX(0.95) + (g ** 2)*floatX(0.05)) for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function(inp, cost, updates=zgup+rg2up,
                                    profile=False, on_unused_input='ignore')
    
    updir = [-T.sqrt(ru2 + 1e-6) / T.sqrt(rg2 + 1e-6) * zg for zg, ru2, rg2 in zip(zipped_grads, running_up2, running_grads2)]
    ru2up = [(ru2, 0.95 * ru2 + 0.05 * (ud ** 2)) for ru2, ud in zip(running_up2, updir)]
    param_up = [(p, p + ud) for p, ud in zip(itemlist(tparams), updir)]

    f_update = theano.function([lr], [], updates=ru2up+param_up, on_unused_input='ignore', profile=False)

    return f_grad_shared, f_update

def adam(lr, tparams, grads, inp, cost):
    gshared = [theano.shared(p.get_value() * 0., name='%s_grad'%k) for k, p in tparams.iteritems()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]

    f_grad_shared = theano.function(inp, cost, updates=gsup)

    lr0 = 0.0002
    b1 = 0.1
    b2 = 0.001
    e = 1e-8

    updates = []

    i = theano.shared(numpy.float32(0.))
    i_t = i + 1.
    fix1 = 1. - b1**(i_t)
    fix2 = 1. - b2**(i_t)
    lr_t = lr0 * (tensor.sqrt(fix2) / fix1)

    for p, g in zip(tparams.values(), gshared):
        m = theano.shared(p.get_value() * 0.)
        v = theano.shared(p.get_value() * 0.)
        m_t = (b1 * g) + ((1. - b1) * m)
        v_t = (b2 * tensor.sqr(g)) + ((1. - b2) * v)
        g_t = m_t / (tensor.sqrt(v_t) + e)
        p_t = p - (lr_t * g_t)
        updates.append((m, m_t))
        updates.append((v, v_t))
        updates.append((p, p_t))
    updates.append((i, i_t))

    f_update = theano.function([lr], [], updates=updates, on_unused_input='ignore')

    return f_grad_shared, f_update

def rmsprop(lr, tparams, grads, inp, cost):
    zipped_grads = [theano.shared(p.get_value() * numpy.float32(0.), name='%s_grad'%k) for k, p in tparams.iteritems()]
    running_grads = [theano.shared(p.get_value() * numpy.float32(0.), name='%s_rgrad'%k) for k, p in tparams.iteritems()]
    running_grads2 = [theano.shared(p.get_value() * numpy.float32(0.), name='%s_rgrad2'%k) for k, p in tparams.iteritems()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rgup = [(rg, floatX(0.95) * rg + floatX(0.05) * g) for rg, g in zip(running_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2)) for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function(inp, cost, updates=zgup+rgup+rg2up, profile=False)

    updir = [theano.shared(p.get_value() * numpy.float32(0.), name='%s_updir'%k) for k, p in tparams.iteritems()]
    updir_new = [(ud, 0.9 * ud - 1e-4 * zg / tensor.sqrt(rg2 - rg ** 2 + 1e-4)) for ud, zg, rg, rg2 in zip(updir, zipped_grads, running_grads, running_grads2)]
    param_up = [(p, p + udn[1]) for p, udn in zip(itemlist(tparams), updir_new)]
    f_update = theano.function([lr], [], updates=updir_new+param_up, on_unused_input='ignore', profile=False)

    return f_grad_shared, f_update

def sgd(lr, tparams, grads, inp, cost):
    gshared = [theano.shared(p.get_value() * numpy.float32(0.), name='%s_grad'%k) for k, p in tparams.iteritems()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]

    f_grad_shared = theano.function(inp, cost, updates=gsup, profile=False)

    pup = [(p, p - lr * g) for p, g in zip(itemlist(tparams), gshared)]
    f_update = theano.function([lr], [], updates=pup, profile=False)

    return f_grad_shared, f_update

def itemlist(tparams):
    return [vv for kk, vv in tparams.iteritems()]
