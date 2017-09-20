"""
My implementation for optimization
"""

import numpy as np
import pickle
import quadprog


"""
Use QP to solve tracking problem:
    \min 0.5*||w||^2 +mu \sum_i \ xi_i
    s.t. w>=0, \ xi_i>=0, w^Tx_i>=1-\ xi_i
"""
def solve_tracking_qp(feats, mu):
    datanum, featnum = feats.shape
    eps = 1e-11
    P = np.zeros((featnum+datanum,featnum+datanum))
    for i in range(featnum): P[i,i] = 1
    for i in range(featnum+datanum): P[i,i]+=eps
    q = np.array([0]*featnum+[mu]*datanum).astype('float')
    G = np.concatenate((-feats, -np.eye(datanum)),axis=1)
    G = np.concatenate((-np.eye(datanum+featnum),G),axis=0)
    h = np.array([0]*(datanum+featnum)+[-1]*datanum).astype('float')
    sol = quadprog_solve_qp(P,q,G=G,h=h)
    w = sol[:featnum]
    return w

### SVM like problem
def quadprog_solve_qp(P, q, G=None, h=None, A=None, b=None):
    qp_G = .5 * (P + P.T)
    qp_a = -q
    if A is not None:
        qp_C = -np.vstack([A, G]).T
        qp_b = -np.hstack([b, h])
        meq = A.shape[0]
    else:  # no equality constraint
        qp_C = -G.T
        qp_b = -h
        meq = 0
    return quadprog.solve_qp(qp_G, qp_a, qp_C, qp_b, meq)[0]

def zero_one_loss(feats, w):
    return np.sum(feats.dot(w)<0)

if __name__=='__main__':
    feats = pickle.load(open('feats_accu.pkl','rb'))
    feats2 = feats[1500:3000,:]
    print feats2.shape
    w = solve_tracking_qp(feats2, 0.01)
    print w
    print np.sum(feats.dot(w)<0)
    
