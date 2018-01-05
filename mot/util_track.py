import sys
import numpy as np
from skimage import io
import cv2

def vis_track(img,bboxes,tar_idxs,codes):
    for i, tar_idx in enumerate(tar_idxs):
        code = codes[tar_idx%len(codes)]
        bb = bboxes[i]
        x1,y1,x2,y2 = bb[0],bb[1],bb[2],bb[3]
        cv2.rectangle(img,(x1,y1),(x2,y2),code,2)
    img = cv2.resize(img,None,fx=0.5,fy=0.5)
    cv2.imshow('track', img)
    cv2.waitKey(-1)

    
def colorcodes(scale=8):
    lst = []
    step = 256/scale
    for i in range(scale):
        for j in range(scale):
            for k in range(scale): lst.append((i*step,j*step,k*step))
    from random import shuffle
    shuffle(lst)
    return lst

"""
adj_mat: lil_matrix or csr_matrix
"""
def get_all_subgraphs(adj_mat):
    num = adj_mat.shape[0]

    nblsts = []
    for i in range(num):
        #nblst = [j for j in range(num) if adj_mat[i,j]==1]
        nblst = adj_mat.getrow(i).nonzero()[1].tolist()
        nblsts.append(nblst)

    clus = [-1]*num
    idx = 0
    for i in range(num):
        if clus[i]!=-1: continue
        clus[i] = idx
        qu,quset,quidx = [i],set([i]),0
        while quidx<len(qu):
            for qi in nblsts[qu[quidx]]:
                if qi not in quset:
                    qu.append(qi)
                    quset.add(qi)
            quidx+=1
        for qi in qu: clus[qi] = idx
        idx+=1
    return clus

def getPatches(img, dets):
    patchlst = []
    for i,det in enumerate(dets):
        x1,y1,x2,y2 = int(det[0]),int(det[1]),int(det[2]),int(det[3])
        patch = img[y1:y2,x1:x2]
        patchlst.append(patchlst)
    return patchlst
    

def yieldMOTDetNGt(detfn, img_dir, gtfn):
    seq_dets = np.loadtxt(detfn,delimiter=',')
    seq_gt = np.loadtxt(gtfn,delimiter=',')
    for frame in range(int(seq_dets[:,0].max())):
        frame += 1
        dets = seq_dets[seq_dets[:,0]==frame,2:7]
        gt = seq_gt[seq_gt[:,0]==frame,:]
        dets[:,2:4] += dets[:,0:2]
        gt[:,4:6] += gt[:,2:4]
        imgfn = img_dir+str(frame).zfill(6)+'.jpg'
        img = io.imread(imgfn)
        patchlst = getPatches(img, dets)
        yield frame, patchlst, img, dets, gt

def yieldMOTDet(fn):
    seq_dets = np.loadtxt(fn,delimiter=',')
    for frame in range(int(seq_dets[:,0].max())):
        frame += 1 #detection and frame numbers begin at 1
        dets = seq_dets[seq_dets[:,0]==frame,2:7]
        dets[:,2:4] += dets[:,0:2] #convert to [x1,y1,w,h] to [x1,y1,x2,y2]
        yield frame, dets

def yieldMOTImgDet(fn, img_dir):
    for frame, dets in yieldMOTDet(fn):
        imgfn = img_dir+str(frame).zfill(6)+'.jpg'
        img = io.imread(imgfn)
        patchlst = []
        for i,det in enumerate(dets):
            x1,y1,x2,y2 = int(det[0]),int(det[1]),int(det[2]),int(det[3])
            patch = img[y1:y2,x1:x2]
            patchlst.append(patchlst)
        yield frame, img, patchlst, dets

def getMotTargetPatches(gtfn, img_dir):
    seq_dets = np.loadtxt(gtfn,delimiter=',').astype('int')
    idset = set(seq_dets[:,1].tolist())
    patchlst_dict = {}
    for id in idset: patchlst_dict[id] = []
    for frame in range(int(seq_dets[:,0].max())):
        frame+=1
        imgfn = img_dir+str(frame).zfill(6)+'.jpg'
        img = io.imread(imgfn)
        data = seq_dets[seq_dets[:,0]==frame,:]
        data[:,4:6] += data[:,2:4]
        for d in data:
            id = d[1]
            x1,y1,x2,y2 = d[2:6]
            patch = img[y1:y2,x1:x2]
            patchlst_dict[id].append(patch)
    return patchlst_dict

def plot_detection(detfn, img_dir):
    for frame, img, patchlst, dets in yieldMOTImgDet(detfn, img_dir):
        for det in dets:
            x1,y1,x2,y2 = int(det[0]),int(det[1]),int(det[2]),int(det[3])
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
        cv2.imshow('det', img)
        cv2.waitKey(-1)
    
if __name__=='__main__':
    gtfn = '/home/tingyaoh/data/MOT/2DMOT2015/train/ADL-Rundle-6/gt/gt.txt'
    detfn = '/home/tingyaoh/data/MOT/2DMOT2015/train/ADL-Rundle-6/det/det.txt'
    img_dir = '/home/tingyaoh/data/MOT/2DMOT2015/train/ADL-Rundle-6/img1/'
