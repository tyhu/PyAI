import sys
sys.path.append('../util')
from util_ml import *
import numpy as np
from sklearn.linear_model import LogisticRegression

### early prediction with latent variable
###  X -- training accumulated features: dims (datanum,featnum,max_time_axis)
###  y -- labels
###  l_tr -- true length of each sequence
def latent_predict(X,y,l_tr,ld):
    maxl = X.shape[2]
    l_ceil = np.ceil(l_tr).astype('int')
    datanum = X.shape[0]
    clf = LogisticRegression(C=10000,penalty='l1')
    #clf = LogisticRegression(C=0.001)
    clf.fit(X[:,:,-1],y)
    iternum = 10
    #print X.shape
    for it in range(iternum):
        Xtrain = None
        ytrain = []
        for idx in range(datanum):
            v = infer(clf,X[idx,:,:].T,y[idx],min(l_ceil[idx],maxl),ld).astype('int')
            X_add = X[idx,:,v==1]
            #print v
            #print X_add.shape
            y_add = [y[idx]]*X_add.shape[0]
            Xtrain = X_add if Xtrain is None else np.concatenate((Xtrain,X_add),axis=0)
            ytrain+=y_add
        clf = LogisticRegression(C=1000,penalty='l1')
        print Xtrain.shape
        clf.fit(Xtrain,ytrain)

    Xtrain = None
    ytrain = []
    clf2 = LogisticRegression(C=1000,penalty='l1')
    for idx in range(datanum):
        l = min(l_ceil[idx],maxl)
        v = infer(clf,X[idx,:,:].T,y[idx],l,ld).astype('int')
        #print v
        for jdx in range(l):
            X_add = X[idx,:,jdx:jdx+1].T
            #print X_add.shape
            y_add = 0 if v[jdx]==0 else y[idx]
            Xtrain = X_add if Xtrain is None else np.concatenate((Xtrain,X_add),axis=0)
            ytrain.append(y_add)

    #print Xtrain.shape
    #print len(ytrain)
    clf2.fit(Xtrain,ytrain)

    #print np.array_equal(Xtrain[:,:],X[:,:,-1])
    #print Xtrain.shape
    #print y
    #print ytrain
    
    return clf2

### infer the hidden variables
### minimizing \sum L_i - \labmda \sum v_i
def infer(clf,X,y,l,ld):
    v = np.zeros(X.shape[0])
    #v = np.zeros(l)
    loss_vec = loss(clf,X,y,l)
    ### exhausted search min objective
    minobj = sys.float_info.max
    minidx = l
    obj = 0
    for idx in reversed(range(l)):
        obj+=loss_vec[idx]-ld
        if obj<minobj:
            minobj = obj
            minidx = idx
    v[minidx:l] = 1
    return v

def loss(clf,X,y,l):
    type_str = str(type(clf))
    class_vec = clf.classes_.tolist()
    #yidx = label_to_index(class_vec,y)
    yidx = class_vec.index(y)
    loss_vec = np.zeros(l)
    if 'LogisticRegression' in type_str:
        loss_mat = clf.predict_proba(X)
        loss_vec = 1-loss_mat[:l,yidx]
    return loss_vec
       

