### Ting-Yao Hu, 2016.06
### adaptive edit distance

import sys
import subprocess

def g2p(s):    
    g2pEXE = '/home2/tingyaoh/flite/flite-2.0.0-release/bin/t2p'
    p = subprocess.Popen([g2pEXE, s], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()
    return out.strip()

class EditDistanceUtil(object):

    #def __init__(self):

    def editDist(self,lst1,lst2):
        m = len(lst1)+1
        n = len(lst2)+1
        d = range(m)
        for idx in range(n-1):
            prev = d[:]
            d[0] = prev[0]+1
            for jdx in range(1,m):
                if lst1[jdx-1]==lst2[idx]:
                    r = prev[jdx-1]
                else: r = prev[jdx-1]+1
                l = d[jdx-1]+1
                u = prev[jdx]+1
                d[jdx] = min(l,u,r)
        return d[-1]
    
    ### for flite output lst
    def phonelstSplit(self,pstr):
        plst = pstr.split()
        vlst, clst = [],[]
        vows = 'aeiou'
        for p in plst:
            if p=='pau': continue
            if any(v in p for v in vows): vlst.append(p)
            else: clst.append(p)
        return vlst, clst


if __name__=='__main__':
    eUtil = EditDistanceUtil()
    #print eUtil.phonelstSplit('pau ae1 p')
    for line in file('student.txt'):
        lst = g2p(line).split()[1:-1]
        print ' '.join(lst)
