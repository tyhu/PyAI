import sys
import math
import gc
from sklearn.preprocessing import normalize
from util_ml import *

from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
#customize distance metric for sklearn kmeans
from sklearn.metrics.pairwise import cosine_similarity
def new_euclidean_distances(X, Y=None, Y_norm_squared=None, squared=False):
    return cosine_similarity(X,Y)
from sklearn.cluster.k_means_ import euclidean_distances 
euclidean_distances = new_euclidean_distances 

#x_accu: accumulated set
#n: current number of samples in the set
#x_new: new sample

#solve submodular optimization problem using greedy algorithm
#score --
#feat --
#budget --
#ld --
def greedySubmodular1(score,labels,budget,ld):
    selectlst = []
    labelset = np.unique(labels).tolist()
    labelHist = np.zeros(len(labelset))

    #First select the one with highest score
    candidateLst = range(len(score))
    idx = np.argmax(score)
    selectlst.append(idx)
    candidateLst.remove(idx)
    obj = score[idx]
    pred_idx = labelset.index(labels[idx])
    labelHist[pred_idx]+=1

    for i in range(budget-1):
        maxobj = obj
        maxidx = -1
        for c in candidateLst:
            pred_idx = labelset.index(labels[c])
            labelHist[pred_idx]+=1
            tmpobj = obj+score[c]+ld*np.sum(np.sqrt(labelHist))
            if tmpobj>maxobj:
                maxobj = tmpobj
                maxidx = c
            labelHist[pred_idx]-=1
        candidateLst.remove(maxidx)
        pred_idx = labelset.index(labels[maxidx])
        labelHist[pred_idx]+=1
        obj = maxobj
        selectlst.append(maxidx)
    #print labelHist
    return selectlst

def greedySubmodular2(score,labels,curLabels,budget,ld):
    selectlst = []
    labelset = np.unique(np.concatenate((curLabels,labels))).tolist()
    labelHist = np.zeros(len(labelset))

    #set current labels
    for lab in curLabels.tolist():
        idx = labelset.index(lab)
        labelHist[idx]+=1

    obj = ld*np.sum(np.sqrt(labelHist))
    candidateLst = range(len(score))
    for i in range(budget):
        maxobj = obj
        maxidx = -1
        for c in candidateLst:
            pred_idx = labelset.index(labels[c])
            labelHist[pred_idx]+=1
            tmpobj = obj+score[c]+ld*np.sum(np.sqrt(labelHist))
            if tmpobj>maxobj:
                maxobj = tmpobj
                maxidx = c
            labelHist[pred_idx]-=1
        candidateLst.remove(maxidx)
        pred_idx = labelset.index(labels[maxidx])
        labelHist[pred_idx]+=1
        obj = maxobj
        selectlst.append(maxidx)
    #print labelHist
    return selectlst

### kNN submodular for 
### this one works
### X here is a csr_matrix
def greedySubmodular3(X,labels,budget):
    labelset = np.unique(labels).tolist()
    labelHist = np.zeros(len(labelset))
    ### set current labels
    for lab in labels.tolist():
        idx = labelset.index(lab)
        labelHist[idx]+=1

    Xnorm = normalize(X, norm='l2', axis=1)
    selectlst = []
    #print labelHist
    #simMat = Xnorm*(Xnorm.T)
    ratio = float(budget)/len(labels)

    ### assign  label distribution
    labdist = []
    for lab in labelset:
        labidxs = [i for i, x in enumerate(labels) if x == lab]
        labselectsize = int(math.floor(len(labidxs)*ratio))+1
        labdist.append(labselectsize)
    labdist = np.array(labdist)
    while np.sum(labdist)>budget:
        idx = np.argmax(labdist)
        labdist[idx]-=1
        

    for labi, lab in enumerate(labelset):
        labidxs = [i for i, x in enumerate(labels) if x == lab]
        X_lab = Xnorm[labidxs]
        simMat = X_lab*(X_lab.T).toarray()
        #simMat = X_lab*(X_lab.T)
        #print simMat.shape
        #labselectsize = int(math.floor(len(labidxs)*ratio))+1
        labselectsize = labdist[labi]
        labcandidate = range(len(labidxs))
        labselect = []
        
        ### find the first one
        sidx = np.argmax(simMat.sum(axis=1))
        labselect.append(sidx)
        labcandidate.remove(sidx)
        obj = np.max(simMat.sum(axis=1))
        ### greedy select
        for idx in range(1,labselectsize):
            maxobj = obj
            maxidx = labcandidate[0]
            for pt in labcandidate:
                labselect.append(pt)
                tmpobj = np.sum(np.max(simMat[labselect],axis=0))
                if tmpobj>maxobj:
                    maxidx = pt
                    maxobj = tmpobj
                labselect.pop()
            labselect.append(maxidx)
            labcandidate.remove(maxidx)
        
        ### consider the label ratio only
        labselect = np.random.permutation(len(labidxs))[:labselectsize].tolist()

        for idx in labselect:
            selectlst.append(labidxs[idx])
        del simMat
    #gc.collect()a
    #print len(selectlst)
    selectlst = np.array(selectlst)[np.random.permutation(len(selectlst))].tolist()
    #print selectlst[:budget]
    return selectlst[:budget]

### modified kNN submodular
def greedySubmodular4(X,y,Xtrain,ytrain,ld,budget):
    labelset = np.unique(y).tolist()
    labelHist = np.zeros(len(labelset))
    ### set current labels
    for lab in y.tolist():
        idx = labelset.index(lab)
        labelHist[idx]+=1
    
    Xnorm = normalize(X, norm='l2', axis=1)
    selectlst = []
    #print labelHist
    #simMat = Xnorm*(Xnorm.T)
    ratio = float(budget)/len(y)

    ### assign  label distribution
    labdist = []
    for lab in labelset:
        labidxs = [i for i, x in enumerate(y) if x == lab]
        labselectsize = int(math.floor(len(labidxs)*ratio))+1
        labdist.append(labselectsize)
    labdist = np.array(labdist)
    while np.sum(labdist)>budget:
        idx = np.argmax(labdist)
        labdist[idx]-=1

    for labi, lab in enumerate(labelset):
        labidxs = [i for i, x in enumerate(y) if x == lab]
        X_lab = Xnorm[labidxs]
        simMat = X_lab*(X_lab.T).toarray()

        trainlabidxs = [i for i, x in enumerate(ytrain) if x==lab]
        #print lab,trainlabidxs
        Xtrain_lab = Xtrain[trainlabidxs]
        simToTrain = X_lab*(Xtrain_lab.T).toarray()

        #print simMat.shape
        #labselectsize = int(math.floor(len(labidxs)*ratio))+1
        labselectsize = labdist[labi]
        labcandidate = range(len(labidxs))
        labselect = []

        obj = -100000
        ### greedy select
        for idx in range(0,labselectsize):
            maxobj = -100000
            maxidx = -1
            for pt in labcandidate:
                labselect.append(pt)
                tmpobj =  np.sum(np.max(simMat[labselect],axis=0)) - ld*np.max(simToTrain[pt])
                if tmpobj>maxobj:
                    maxidx = pt
                    maxobj = tmpobj
                labselect.pop()
            labselect.append(maxidx)
            #print maxidx
            labcandidate.remove(maxidx)
        
        ### consider the label ratio only
        #labselect = np.random.permutation(len(labidxs))[:labselectsize].tolist()

        for idx in labselect:
            selectlst.append(labidxs[idx])
        del simMat
    #gc.collect()
    #selectlst = np.array(selectlst)[np.random.permutation(len(selectlst))].tolist()
    #print selectlst[:budget]
    return selectlst[:budget]

def unsupSubmodular(X, budget):
    Xnorm = normalize(X, norm='l2', axis=1)
    selectlst = []
    simMat = (Xnorm*Xnorm.T).toarray()
    datanum = X.shape[0]
    candidate = range(datanum)
    select = []
    sidx = np.argmax(simMat.sum(axis=1))
    select.append(sidx)
    candidate.remove(sidx)
    obj = np.max(simMat.sum(axis=1))

    """
    for idx in range(1,budget):
        minobj = 1
        minidx = -1
        for pt in candidate:
            obj = np.max(simMat[select],axis=0)[pt]
            if obj<minobj:
                minidx = pt
                minobj = obj
        print minobj
        select.append(minidx)
        candidate.remove(minidx)
    """
    
    
    for idx in range(1,budget):
        maxobj = obj
        maxidx = -1
        for pt in candidate:
            select.append(pt)
            tmpobj =  np.sum(np.max(simMat[select],axis=0))
            #print tmpobj,obj
            if tmpobj>maxobj:
                maxidx = pt
                maxobj = tmpobj
            select.pop()
        select.append(maxidx)
        candidate.remove(maxidx)
        obj = np.sum(np.max(simMat[select],axis=0))
     
    return select
