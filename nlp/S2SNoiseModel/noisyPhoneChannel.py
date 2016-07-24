### Ting-Yao Hu, 2016.06
###
import sys
import numpy as np
import cPickle as pickle

class NoisyChannelModel(object):

    ### initial phoneme generative distribution
    ### s -- phone set (list)
    def initPhoneDist(self,s):
        l = len(s)
        self.distMat = 0.9*np.eye(l)+0.1*np.ones((l,l))/l
        self.distMat[:,0] = 0.01*np.ones(l)
        self.distMat[0,0] = 0.99
        self.idxDict = {}
        for idx,p in enumerate(s):
            self.idxDict[p] = idx
        #print self.distMat
    
    ### p1,p2,p3... => dummy, p1, dummy, p2, dummy, p3...
    def extendTarget(self,tseq):
        outlst = []
        for p in tseq:
            outlst.append('NULL')
            outlst.append(p)
        outlst.append('NULL')
        return outlst

    ### return the alignment with highest probability
    ### seq -- observed phone sequence
    ### tseq -- target sequence
    def bestAlignment(self,seq,tseq):
        tseq = self.extendTarget(tseq)
        l, lt = len(seq), len(tseq)
        ll, backtrack, dp = np.zeros((l,lt)), np.zeros((l,lt),dtype='int'), np.zeros((l,lt))
        distMat,idxDict = self.distMat, self.idxDict

        ### llmatrix
        for i, pt in enumerate(tseq):
            for j, p in enumerate(seq):
                ll[j][i] = np.log(distMat[idxDict[p]][idxDict[pt]])

        ### dp and backtract
        wthres = 4
        nullprob = 0
        for j, pt in enumerate(tseq):
            dp[0][j] = ll[0][j]+nullprob
            backtrack[0][j] = j-1
            nullprob+=np.log(distMat[idxDict['NULL']][idxDict[pt]])

        for j, pt in enumerate(tseq):
            if j==0:
                nullprob = 0
                for i in range(len(seq)):
                    dp[i][0] = ll[i][0]+nullprob
                    nullprob = dp[i][0]
                continue
            wsize = min(wthres,j)
            if pt!='NULL': tseqseg, segidx = tseq[j-wsize:j], range(j-wsize,j)
            else: tseqseg, segidx = tseq[j-wsize:j+1], range(j-wsize,j+1)
            tseqseg,segidx = list(reversed(tseqseg)), list(reversed(segidx))
            for i, p in enumerate(seq):
                if i==0: continue
                nullprob = 0
                maxprob = -100000000
                #print i,j,segidx
                for k, pk in zip(segidx,tseqseg):
                    prob = nullprob + dp[i-1][k] + ll[i][j]
                    if prob>maxprob:
                        maxidx = k
                        maxprob = prob
                    nullprob+=np.log(distMat[idxDict['NULL']][idxDict[pk]])
                dp[i][j] = maxprob
                backtrack[i][j] = maxidx
        ### dp, fix tail (probability that all phone in tail generate NULL)
        nullprob = 0
        for j in reversed(range(len(tseq))):
            dp[-1][j]+=nullprob
            pt = tseq[j]
            nullprob+=np.log(distMat[idxDict['NULL']][idxDict[pt]])

        maxend = np.argmax(dp[-1,:])
        score = np.max(dp[-1,:])
        align = [maxend]
        for i in reversed(range(len(seq)-1)):
            align.append(backtrack[i+1][align[-1]])
        align.reverse()
        #print ll
        #print dp
        #print backtrack
        #print tseq
        #print seq
        return align,score

    def scoreList(self,seqs,tseq):
        scorelst = []
        for seq in seqs:
            _,score = self.bestAlignment(seq,tseq)
            scorelst.append(score)
        return scorelst

    def enumerateAlign(self,seq,start,seqt,startt):
        l, lt = len(seq), len(seqt)
        if startt==lt: return [[]]
        if abs(lt-startt-l+start)>1: return []
        outlst = [[-1]+lst for lst in self.enumerateAlign(seq,start,seqt,startt+1)]
        for idx in range(l-start):
            outlst+=[[start+idx]+lst for lst in self.enumerateAlign(seq,start+idx+1,seqt,startt+1)]
        return outlst
            
    def likelihood(self,seq,seqt,align):
        prob = 1
        for idx,p in enumerate(seqt):
            alignidx = align[idx]
            gp = 'NULL' if alignidx==-1 else seq[alignidx]
            prob*=self.distMat[self.idxDict[p]][self.idxDict[gp]]
        diff = int(abs(len(seq)-(len(align)-align.count(-1))))
        if diff>3: diff=3
        return prob*self.dummyDist[diff]
        #return prob

    def EMTrain(self,tseqlst,seqlst,alignlst):
        l = len(self.idxDict)
        idxDict = self.idxDict
        stat = np.zeros((l,l))
        for tseq, seq, align in zip(tseqlst,seqlst,alignlst):
            tseq2 = self.extendTarget(tseq)
            for p, idx in zip(seq,align):
                i,j = idxDict[p],idxDict[tseq2[idx]]
                stat[i,j]+=1
        s = np.sum(stat,axis=0)
        for i in range(l):
            if s[i]>0: stat[:,i]/=s[i]
        #print stat.tolist()
        alpha = 0.05
        self.distMat = self.distMat*(1-alpha)+stat*alpha

if __name__=='__main__':
    phonelst = [l.strip() for l in file('lex2','r')]
    ncm = NoisyChannelModel()
    ncm.initPhoneDist(phonelst)
    seqt = 'f ax t l'.split()
    seq = 'f t ch'.split()
    #seqt = 'ax ae1 f t'.split()
    #seq = 'ax ae1 ow1 t'.split()
    lst =  ncm.enumerateAlign(seq,0,seqt,0)
    #for l in lst:
    #    print l,ncm.likelihood(seq,seqt,l)
    print ncm.bestAlignment(seq,seqt)
    
    seqlst,tseqlst,alignlst = [],[],[]
    targetlst = pickle.load(open('targetlst.pkl'))
    spkr_trans_lst = pickle.load(open('spkr_trans_lst.pkl'))
    for translst in spkr_trans_lst:
        for idx,name in enumerate(translst):
            print name[0]
            print targetlst[idx][0]
            print name[1] +' ### '+ targetlst[idx][1]
            seq1 = name[1].split()[1:-1]
            seq2 = targetlst[idx][1].split()[1:-1]
            if len(seq1)==0 or len(seq2)==0: continue
            align = ncm.bestAlignment(seq1,seq2)
            tseqlst.append(seq2)
            seqlst.append(seq1)
            alignlst.append(align)
    #ncm.EMTrain(tseqlst,seqlst,alignlst)
    
