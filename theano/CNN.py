import numpy as np
import theano
import theano.tensor as T
from util_theano import *
from compact_cnn import *
from sklearn.metrics import accuracy_score

class CNNClassifier(object):
    def __init__(self, lr=0.01, momentum=0.9, ld2=0.1):
        x = T.tensor4()
        t = T.matrix()
        y, c, params,p_y_given_x = self.buildmodel(x,t)

        ### theano function
        updates = updatefunc(c, params, lr, momentum)
        self.train = theano.function([x, t], c, updates=updates)
        self.predict_ = theano.function([x], y)
        self.prob_ = theano.function([x],p_y_given_x)

    def fit(self, X, y, batchsize=32):
        X = floatX(X)
        for i in range(50):
            totalcost = 0
            print "iteration %d" % (i + 1)
            for start in range(0, len(X), batchsize):
                x_batch = X[start:start + batchsize]
                y_batch = y[start:start + batchsize]
                cost = self.train(x_batch, y_batch)
                totalcost+=cost
            print 'cost: ',totalcost

    def predict(self,X):
        datanum = X.shape[0]
        ypred = []
        for i in range(datanum):
            y = self.predict_(X[i:i+1,:,:,:])
            ypred.append(y[0])
        return ypred

    def prob(self,X):
        return self.prob_(X)

    def buildmodel(self,x,t):
        rng = np.random.RandomState(12345)
        w_c1 = init_weights_rng((30, 3, 5, 5),rng)
        b_c1 = init_weights_rng((30,),rng)
        w_c2 = init_weights_rng((30, 30, 5, 5),rng)
        b_c2 = init_weights_rng((30,),rng)
        w_c3 = init_weights_rng((20, 30, 5, 5),rng)
        b_c3 = init_weights_rng((20,),rng)
        w_h = init_weights_rng((20 * 4 * 4, 200),rng)
        b_h = init_weights_rng((200,),rng)
        w_o = init_weights_rng((200, 2),rng)
        b_o = init_weights_rng((2,),rng)

        c1 = addConvLayer(x,w_c1,b_c1,border_mode=2)
        p1 = addPoolLayer(c1,psize=(2,2))
        c2 = addConvLayer(p1,w_c2,b_c2,border_mode=2)
        p2 = addPoolLayer(c2,psize=(2,2))
        c3 = addConvLayer(p2,w_c3,b_c3,border_mode=2)
        p3 = addPoolLayer(c3,psize=(2,2))
        flat = p3.flatten(2)
        o4 = addFullLayer(flat,w_h,b_h)
        p_y_given_x = addSoftmaxLayer(o4,w_o,b_o)
        y = T.argmax(p_y_given_x, axis=1)

        params = [w_c1, b_c1, w_c2, b_c2, w_c3, b_c3, w_h, b_h, w_o, b_o]
        l2 = l2norm(params)
        ld = 0.001
        c = negll(p_y_given_x, t)+ld*l2

        return y, c, params, p_y_given_x
    """
    def buildmodel2(self,x,t):
        rng = np.random.RandomState(12345)
        w_c1 = init_weights_rng((30, 3, 3, 3),rng)
        b_c1 = init_weights_rng((30,),rng)
        w_c2 = init_weights_rng((20, 30, 3, 3),rng)
        b_c2 = init_weights_rng((20,),rng)
        w_h3 = init_weights_rng((20 * 8 * 8, 200),rng)
    """
