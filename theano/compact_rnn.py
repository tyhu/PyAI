import numpy as np
import theano
import theano.tensor as T
from myutil import *

def getLSTMParams(xdim,hdim,rng):

    W_xi = init_weights_rng((xdim,hdim),rng)
    W_hi = init_weights_rng((hdim,hdim),rng)
    W_ci = init_weights_rng((hdim,hdim),rng)

    W_xf = init_weights_rng((xdim,hdim),rng)
    W_hf = init_weights_rng((hdim,hdim),rng)
    W_cf = init_weights_rng((hdim,hdim),rng)

    W_xo = init_weights_rng((xdim,hdim),rng)
    W_ho = init_weights_rng((hdim,hdim),rng)
    W_co = init_weights_rng((hdim,hdim),rng)

    W_xc = init_weights_rng((xdim,hdim),rng)
    W_hc = init_weights_rng((hdim,hdim),rng)
    
    b_i = init_weights_rng((hdim,),rng)
    b_f = init_weights_rng((hdim,),rng)
    b_o = init_weights_rng((hdim,),rng)
    b_c = init_weights_rng((hdim,),rng)

    return [W_xi,W_hi,W_ci,W_xf,W_hf,W_cf,W_xo,W_ho,W_co,W_xc,W_hc,b_i,b_f,b_o,b_c]

def lstm_activation(x_t, h_tm1, c_tm1, params):
    W_xi,W_hi,W_ci,W_xf,W_hf,W_cf,W_xo,W_ho,W_co,W_xc,W_hc,b_i,b_f,b_o,b_c = params[:]
    i_t = T.nnet.sigmoid(T.dot(x_t,W_xi) + \
        T.dot(h_tm1,W_hi) + \
        T.dot(c_tm1,W_ci) + \
        b_i)

    f_t = T.nnet.sigmoid(T.dot(x_t,W_xf) + \
        T.dot(h_tm1,W_hf) + \
        T.dot(c_tm1,W_cf) + \
        b_i)

    c_t = f_t * c_tm1 + i_t * \
        T.tanh(T.dot(x_t,W_xc) + \
            T.dot(h_tm1,W_hc) + \
            b_c)

    o_t = T.nnet.sigmoid(T.dot(x_t,W_xo) + \
        T.dot(h_tm1,W_ho) + \
        T.dot(c_t,W_co) + \
        b_i)
    h_t = o_t * T.tanh(c_t)

    return h_t, c_t

### rnn_params -- [W_xh, W_hh, W_hy, b_h, b_y]
### rnn_const -- h0 (rnn), [h0,c0] (lstm)
def addRNNLayer(X,rnn_params,rnn_const,act_mode='rnn',lstm_params=[]):
    W_xh, W_hh, W_hy, b_h, b_y = rnn_params
    if act_mode is ' lstm': h0,c0 = rnn_const
    else:
        h0 = rnn_const
        c0 = h0
    
    def recurrent_fn(x_t, h_tm1, c_tm1 = None):
        if act_mode is 'lstm':
            h_t, c_t = lstm_activation(x_t, h_tm1, c_tm1, lstm_params)
        else:
            h_t = self.activation(T.dot(x_t, self.W_xh) + \
                T.dot(x_tm1, self.W_hh) + \
                self.b_h)
            c_t = h_t
        y_t = T.dot(h_t, self.W_hy) + self.b_y

    [h,c,y_pred], _ = theano.scan(recurrent_fn, sequences=X, outputs_info = [h0, c0, None])

    return y_pred
