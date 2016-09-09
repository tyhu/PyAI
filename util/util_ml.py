#utility functions for machine learning, (for numpy, sklearn and Theano)
#Ting-Yao Hu, 2015.07

import numpy as np
from scipy.sparse import csr_matrix
import cPickle as pickle
import scipy.io

def printtest():
    print 'test it!'

def lib2ScipySparse(fn,featDim=None):
    infile = file(fn,'r')
    label, data, rowidx, colidx = [],[],[],[]
    for idx, line in enumerate(infile):
        lst = line.split()
        label.append(lst[0])
        for jdx in range(1,len(lst)):
            l = lst[jdx].split(':')
            data.append(float(l[1]))
            rowidx.append(idx)
            colidx.append(int(l[0]))
    if featDim is None:
        feat = csr_matrix((data,(rowidx,colidx)),shape=(rowidx[-1]+1,max(colidx)+1))
        featDim = max(colidx)+1
    else:
        feat = csr_matrix((data,(rowidx,colidx)),shape=(rowidx[-1]+1,featDim))
    return feat,np.array(label),featDim

def lib2pkl(infn,featfn,labfn,labtype = 'classification'):
    X, y = lib2nparray(infn,labtype)
    pickle.dump(X,open(featfn,'wb'))
    pickle.dump(y,open(labfn,'wb'))

def lib2nparray(infn,labtype):
    lines = [ line.strip() for line in file(infn,'r') ]
    datanum = len(lines)
    y = np.zeros(datanum)
    featnum = 0
    for line in lines:
        featnum = max(int(line.split()[-1].split(':')[0]),featnum)

    X = np.zeros((datanum,featnum))
    for idx in range(len(lines)):
        lst = lines[idx].split()
        if labtype=='regression': y[idx] = float(lst[0])
        else: y[idx] = int(lst[0])
        for jdx in range(len(lst)-1):
            featpair = lst[jdx+1].split(':')
            X[idx, int(featpair[0])-1] = float(featpair[1])

    return X, y

def ListKFold(X,y,k=5):
    alllst = range(len(X))
    foldsize = int(len(X)/k)
    for idx in range(k):
        Xtrain, Xtest = [],[]
        testlst = range(idx*foldsize,idx*foldsize+foldsize)
        trainlst = [i for i in alllst if i not in testlst]
        for i in testlst: Xtest.append(X[i])
        for i in trainlst: Xtrain.append(X[i])
        ytrain = np.delete(y,testlst,0)
        ytest = y[testlst]
        yield Xtrain, ytrain, Xtest, ytest

def KFold(X,y,k=5):
    foldsize = int(X.shape[0]/k)
    for idx in range(k):
        testlst = range(idx*foldsize,idx*foldsize+foldsize)
        Xtrain = np.delete(X,testlst,0)
        ytrain = np.delete(y,testlst,0)
        Xtest = X[testlst]
        ytest = y[testlst]
        yield Xtrain, ytrain, Xtest, ytest


def KFold_withl(X,y,l,k=5):
    foldsize = int(X.shape[0]/k)
    for idx in range(k):
        testlst = range(idx*foldsize,idx*foldsize+foldsize)
        Xtrain,ytrain,ltrain = np.delete(X,testlst,0),np.delete(y,testlst,0),np.delete(l,testlst,0)
        Xtest,ytest,ltest = X[testlst],y[testlst],l[testlst]
        yield Xtrain,ytrain,ltrain,Xtest,ytest,ltest

def Batch_KFold_withl(X,y,l,bids,k=5):
    bidset = list(set(bids))
    for idx in range(k):
        bid_fold = []
        for jdx in range(len(bidset)):
            if jdx%k==idx: bid_fold.append(bidset[jdx])
        testlst = []
        for jdx in range(X.shape[0]):
            if bids[jdx] in bid_fold: testlst.append(jdx)
        Xtrain,ytrain,ltrain = np.delete(X,testlst,0),np.delete(y,testlst,0),np.delete(l,testlst,0)
        Xtest,ytest,ltest = X[testlst],y[testlst],l[testlst]
        yield Xtrain,ytrain,ltrain,Xtest,ytest,ltest


def LeaveOneOut(X,y):
    datanum, featnum = X.shape
    for idx in range(datanum):
        Xtrain = np.delete(X,idx,0)
        ytrain = np.delete(y,idx,0)
        Xtest = X[idx]
        ytest = y[idx]
        yield Xtrain, ytrain, Xtest, ytest

def NNFormLabel(lab,catogories=None):
    if catogories is None: catogories = np.unique(lab)
    catnum = catogories.shape[0]
    datanum = lab.shape[0]
    nn_lab = np.zeros((datanum,catnum), dtype=int)

    for idx in range(datanum):
        for jdx in range(catnum):
            if lab[idx]==catogories[jdx]: nn_lab[idx, jdx] = 1
    return nn_lab, catogories

def NNLabel2NormLabel(lab_nn, cat):
    datanum, catnum = lab_nn.shape
    lab = np.zeros(datanum)
    for idx in range(datanum):
        for jdx in range(catnum):
            if lab_nn[idx, jdx]==1: cat[jdx]
    return lab

def accuracy(ypred,ytest):
    datanum = ypred.shape[0]
    return float(np.logical_and(ypred,ytest).sum())/datanum

def unweighted_acc(confMat):
    acc = []
    for idx in range(len(confMat)):
        acc.append(float(confMat[idx][idx])/sum(confMat[idx]))
    return sum(acc)/len(acc)
    

#zero mean uni-variance
def normalization(X, meanv = None, var = None):
    train = False
    if meanv is None:
        meanv = X.mean(axis = 0)
        train = True
    if var is None:
        var = X.var(axis = 0)
        var[var==0] = 1
    X_norm = (X - meanv)/var
    if train: return X_norm, meanv, var
    else: return X_norm

def RandomPerm(Xtrain,ytrain):
    per = np.random.permutation(Xtrain.shape[0])
    Xtrain_s = Xtrain[per]
    ytrain_s = ytrain[per]
    return Xtrain_s, ytrain_s

# for list
def ListRandomPerm(Xtrain,ytrain):
    per = np.random.permutation(len(Xtrain))
    Xtrain_s, ytrain_s = [], []
    for i in per.tolist():
        Xtrain_s.append(Xtrain[i])
        ytrain_s.append(ytrain[i])
    return Xtrain_s, np.array(ytrain_s)
    

def random_perm2file(infn,outfn):
    lst = []
    for line in file(infn,'r'):
        lst.append(line.strip())
    idxs = np.random.permutation(len(lst))
    outfile = file(outfn,'w')
    for i in idxs: outfile.write(lst[idxs[i]]+'\n')
    outfile.close()

### map the label name array (ytrain, ytest)
def label_to_index(class_lst,y):
    return np.array([class_lst.index(t) for t in y.tolist()])
