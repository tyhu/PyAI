import numpy as np
import theano
import theano.tensor as T

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

### params -- list of parameters
def l2norm(params):
    l2 = (params[0]**2).sum()
    for param in params[1:]:
        l2 = l2+(param**2).sum()
    return l2

