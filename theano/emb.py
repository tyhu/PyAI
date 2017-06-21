import sys
import numpy as np
import theano
import theano.tensor as T
from util_theano import *
from compact_cnn import *
import cPickle as pickle

"""
deep multiview embedding
x1,x2: feature in view x, y1,y2: feature in view y
bidirectional ranking constraint
  d(x1,y1) + m < d(x1,y2)
  d(x1,y1) + m < d(x2,y1)
"""
def buildDeepMEmb(x1,x2,y1,y2,m):
    
    ### build net work
    h1x1 = addFullLayer(x1, W_h1x, b_h1x)
    h1x2 = addFullLayer(x2, W_h1x, h_h1x)
    
    h2x1 = addFullLinearLayer(h1x1, W_h2x, b_h2x)
    h2x2 = addFullLinearLayer(h1x2, W_h2x, b_h2x)

    bnx1, mean_x, var_x = addFullBNLayerTrain(h2x1, gamma_x, beta_x)
    bnx2, mean_x, var_x = addFullBNLayerTrain(h2x2, gamma_x, beta_x, mean=mean_x, var=var_x)

    l2x1 = l2normalize(bnx1)
    l2x2 = l2normalize(bnx2)

    h1y1 = addFullLayer(y1, W_h1y, b_h1y)
    h1y2 = addFullLayer(y2, W_h1y, h_h1y)
    
    h2y1 = addFullLinearLayer(h1y1, W_h2y, b_h2y)
    h2y2 = addFullLinearLayer(h1y2, W_h2y, b_h2y)

    bny1, mean_y, var_y = addFullBNLayerTrain(h2y1, gamma_y, beta_y)
    bny2, mean_y, var_y = addFullBNLayerTrain(h2y2, gamma_y, beta_y, mean=mean_y, var=var_y)

    l2y1 = l2normalize(bny1)
    l2y2 = l2normalize(bny2)

    ### cost
    dx1y1 = T.sum((l2x1-l2y1)**2)

    return o1,o2,cost,param
