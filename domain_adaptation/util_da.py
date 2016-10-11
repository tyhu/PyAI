import sys
import numpy as np
from scipy.linalg import sqrtm, inv
from numpy.matlib import repmat

'''
ref 
'''
def CORAL(Ds,Dt):
    featnum = Ds.shape[1]
    Cs = np.cov(Ds.T)+np.eye(featnum)
    Ct = np.cov(Dt.T)+np.eye(featnum)
    Ds = Ds.dot(inv(sqrtm(Cs)))
    return Ds.dot(sqrtm(Ct))

'''
X -- data matrix 
p -- dropout rate
'''
def denoiseAutoEncoder(X, p):
    datanum,featnum = X.shape
    Xp = np.concatenate((X,np.ones((datanum,1)) ),axis=1)
    S = Xp.T.dot(Xp)
    EQ = np.zeros_like(S)
    q = [p]*featnum
    q.append(1)
    q = np.array(q)
    EQ = S * (q.dot(q.T))
    EP = S * repmat(q,featnum+1,1)
    W = EP[:-1,:].dot(inv(EQ+1e-5*np.eye(featnum+1)))
    return W



