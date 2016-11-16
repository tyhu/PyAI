
import numpy as np
import theano
import theano.tensor as T
from util_theano import *

class DNN(object):

    def __init__(self, xdim, ydim, hdims, activation='sigmoid',lr=0.0001,momentum=0.9,ld2=0.001):
