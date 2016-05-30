### Ting-Yao Hu, 2016.05

import sys
import os
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
sys.path.append('../util')
from util_ml import *
import cPickle as pickle

def conf_score(Xtest,ypred,clf):
    labset = clf.classes_.tolist()
    type_str = str(type(clf))
    scorelst = []
    for idx in range(len(ypred)):
        lidx = labset.index(ypred[idx])
        if 'LogisticRegression' in type_str:
            score = np.abs(clf.predict_proba(Xtest[idx:idx+1,:])[:,lidx]-0.5)[0]
        elif 'SVC' in type_str:
            score = np.abs(clf.decision_function(Xtest[idx:idx+1,:])[0])
        else: score = 0
        scorelst.append(score)
    return np.array(scorelst)

### 
def statefunc(pred,score,t,l,hist):
    endbool = l<t
    return (pred,np.digitize(score,hist).tolist(),t,endbool)

### calculate the histogram of confidence score
### X,y -- feature(sequence) and label 
def score_hist(X,y,l,clf):
    datanum, featnum, maxl = X.shape
    for Xtrain, ytrain, ltrain, Xtest, ytest, ltest in KFold_withl(X,y,l,5):
        scorelst = []
        for idx in range(maxl):
            clf.fit(Xtrain[:,:,idx],ytrain)
            ypred = clf.predict(Xtest[:,:,idx])
            yscores = conf_score(Xtest[:,:,idx],ypred,clf).tolist()
            scorelst+=yscores
    scores = np.array(scorelst)
    #return sorted([np.percentile(scores, 25),np.percentile(scores, 50),np.percentile(scores, 75)])
    return sorted([np.percentile(scores, 33),np.percentile(scores, 66)])

### feature extraction for reinforcement learning in early prediction
### X -- dim: (datanum, featnum, timesteps)
### y -- label
### l -- sequence length
### clf -- classifier (sklearn)
### hist -- confidence score histogram bin
### stepcost --
def rl_feature(X,y,l,clf,hist,stepcost):
    historylst = []
    datanum, featnum, maxl = X.shape
    for Xtrain, ytrain, ltrain, Xtest, ytest, ltest in KFold_withl(X,y,l,5):
        predlst, scorelst = [],[]
        for idx in range(maxl):
            clf.fit(Xtrain[:,:,idx],ytrain)
            ypred = clf.predict(Xtest[:,:,idx])
            yscores = conf_score(Xtest[:,:,idx],ypred,clf)
            predlst.append(ypred)
            scorelst.append(yscores)
        tsdatanum = Xtest.shape[0]
        for idx in range(tsdatanum):
            statelst = []
            for jdx in range(maxl):
                statelst.append(statefunc(predlst[jdx][idx],scorelst[jdx][idx],jdx+1,ltest[idx],hist))
            for jdx in range(maxl-1):
                r = 1 if predlst[jdx][idx]==ytest[idx] else 0
                state1,state2 = statelst[jdx],statelst[jdx+1]
                endbool = state1[-1]
                if not endbool:
                    historylst.append((state1,'y',r,None))
                    historylst.append((state1,'n',stepcost,state2))
                else:
                    historylst.append((state1,'y',r,None))
                    break
    return historylst

def rl_feature_test(X,clf_lst,l,hist):
    statelst = []
    featnum, _ = X.shape
    lint = min(int(np.ceil(l)),10)
    for idx in range(lint):
        ypred = clf_lst[idx].predict(X[:,idx:idx+1].T)
        yscore = conf_score(X[:,idx:idx+1].T,ypred,clf_lst[idx])
        statelst.append(statefunc(ypred[0],yscore[0],idx+1,l,hist))
    return statelst
    
                
if __name__=='__main__':
    testXfn = '/home2/tingyaoh/sentiment/MOUD/multimodal_feat/audio_seq.pkl'
    testyfn = '/home2/tingyaoh/sentiment/MOUD/multimodal_feat/lab.pkl'
    testlfn = '/home2/tingyaoh/sentiment/MOUD/multimodal_feat/length.pkl'

    X = pickle.load(open(testXfn))
    y = pickle.load(open(testyfn))
    l = pickle.load(open(testlfn))
    clf = LogisticRegression(C=0.01)
    hist = score_hist(X,y,l,clf)
    historylst = rl_feature(X,y,l,clf,hist,0.1)
    print historylst
