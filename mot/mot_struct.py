'''
structure prediction for MOT decision making
'''
import sys
import numpy as np
from util_track import *
from mot_tracker import *
from sklearn.utils.linear_assignment_ import linear_assignment


'''
five types of actions: drop detection, initialize target, match target with detection, report missing target, delete target
'''
def extract_action_sequence(gtfn, detfn):
    seq_targets = np.loadtxt(gtfn,delimiter=',').astype('int')
    seq_dets = np.loadtxt(detfn,delimiter=',').astype('int')
    framenum = int(seq_dets[:,0].max())
    seen_tar_idxs = []
    last_state = None
    for fidx in range(framenum):
        fidx+=1
        dets = seq_dets[seq_dets[:,0]==fidx,:]
        dets[:,4:6] += dets[:,2:4]
        tars = seq_targets[seq_targets[:,0]==fidx,:]
        tars[:,4:6] += tars[:,2:4]
        dnum, tnum = dets.shape[0], tars.shape[0]

        ### last frame info
        if last_state is not None:
            last_tars = seq_targets[seq_targets[:,0]==fidx-1,:]
            last_tars[:,4:6] += last_tars[:,2:4]

        """
        get all decisions
        """
        iou_mat = np.zeros((dnum, tnum))
        for i in range(dnum):
            for j in range(tnum):
                iou_mat[i,j] = iou(dets[i,2:6],tars[j,2:6])
        mi = linear_assignment(-iou_mat)
        tars_state = {}
        for i in range(tnum):
            if i not in mi[:,1]: tars_state[tars[i,1]] = []
            else:
                tars_state[tars[i,1]] = dets[mi[mi[:,1]==i,0],:]

        
        droplist = []
        for i in range(dnum):
            if i not in mi[:,0]: droplist.append(i)
        missinglist = []
        for i in range(tnum):
            if i not in mi[:,1]: missinglist.append(i)
        deletelist = []
        if last_state is not None:
            for idx in last_tars[:,1]:
                if idx not in tars[:,1]: deletelist.append(idx)
        initiallist = []
        for i in range(tnum):
            if tars[i,1] not in seen_tar_idxs and i in mi[:,1]:
                initiallist.append(i)
                seen_tar_idxs.append(tars[i,1])

        ### describe this set of decisions
        vec = represent_decisions(last_state, tars_state, dets, droplist, missinglist, deletelist,initiallist)
    
        last_state = tars_state

"""
represent decisions as a matrix
"""
def represent_decisions(last_state, tars_state, dets, droplist, missinglist, deletelist,initiallist):
    print tars_state
    
if __name__=='__main__':
    gtfn = '/home/tingyaoh/data/MOT/2DMOT2015/train/ETH-Bahnhof/gt/gt.txt'
    detfn = '/home/tingyaoh/data/MOT/2DMOT2015/train/ETH-Bahnhof/det/det.txt'
    img_dir = '/home/tingyaoh/data/MOT/2DMOT2015/train/ETH-Bahnhof/img1/'
    extract_action_sequence(gtfn, detfn)
