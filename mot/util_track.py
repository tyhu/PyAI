import sys
import numpy as np
from skimage import io
import cv2

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
