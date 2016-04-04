### logistic regression
### Ting-Yao Hu, 2016.03

import numpy as np 
import sys

class logistic_regression(object):
    def __init__(self,featdim,labnum,C=1):
        #self.W = 0.1*np.random.rand(featdim,labnum)
        self.W = 1*np.random.rand(featdim,labnum)
        self.C = C
   
    def softmax(self,mat):
        s = mat.sum(axis=1)
        for idx in xrange(mat.shape[0]):
            mat[idx]/=s[idx]
        #print mat
        return mat

    ### objective function (regularized log likelihood)
    def obj(self,X,y):
        v1 = np.sum(np.exp(X.dot(self.W)),axis=1)
        v2 = np.sum(X.dot(self.W)*y,axis=1)
        #print v1.shape
        #print v2.shape
        obj_value = np.sum(v2-np.log(v1))-np.sum(self.W**2)*self.C/2
        return obj_value

    def SGD(self,X,y,lr=0.01):
        pred = self.softmax(np.exp(X.dot(self.W)))
        dW = X.T.dot(y-pred)-self.C*self.W
        self.W = self.W + lr*dW

    def predict(self,X):
        pred = self.softmax(np.exp(X.dot(self.W)))
        y = np.zeros_like(pred)
        for idx in xrange(pred.shape[0]):
            y[idx,np.argmax(pred[idx])]=1
        return y

    ### conditional probability distribution of labels
    def scores(self,X):
        return self.softmax(np.exp(X.dot(self.W)))
        

if __name__=='__main__':
    clf = logistic_regression(C=0.1)

