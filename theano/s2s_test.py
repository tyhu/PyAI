import sys
from lstm import *
import re

chars = 'abcdefghijklmnopqrstuvwxyz'
char_idx_dict = dict([(c,i) for i,c in enumerate(chars)])

def read_samples():
    infn = 'cmudict_SPHINX_40'
    lst = []
    for line in file(infn):
        l = line.split()[0].lower()
        mp = re.match(r'^[A-Za-z]+$',l)
        if mp is not None and len(l)>3:
            lst.append(l)
    return lst

def word2matrix(wd):
    wd = wd.lower()
    l = len(wd)
    mat = np.zeros((l,1,26))
    for i,c in enumerate(wd): mat[i,0,char_idx_dict[c]] = 1
    return mat.astype('float32')

def test():
    read_samples()
    
    w1 = word2matrix('apple')

    X = T.tensor3('X',dtype='float32')
    nsteps, nsample, _ = X.shape
    shape = [26,10]
    f_cost,f_train = build_rnnae(X,shape)

    print 'training with dictionary'
    trainlst = read_samples()
    for i in range(10):
        totalc=0
        for l in trainlst[:10000]:
            w1 = word2matrix(l)
            c = f_cost(w1)
            f_train(0.01)
            totalc+=c
        print 'total cost:',totalc
    
 
if __name__=='__main__':
    test()
