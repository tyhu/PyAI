"""
Multiple Object Tracking Framework
"""
import sys
import numpy as np
from sklearn.utils.linear_assignment_ import linear_assignment

import cv2
hog = cv2.HOGDescriptor()

def colorcodes(scale=4):
    lst = []
    step = 256/scale
    for i in range(scale):
        for j in range(scale):
            for k in range(scale): lst.append((i*step,j*step,k*step))
    return lst
        
def hog_sim(p1,p2):
    h1,h2 = hog.compute(p1), hog.compute(p2)
    return np.mean((h1-h2)**2)

def color_sim(p1,p2):
    h10,h11,h12 = np.histogram(p1[:,:,0],256,[0,256]),np.histogram(p1[:,:,1],256,[0,256]),np.histogram(p1[:,:,2],256,[0,256])
    h20,h21,h22 = np.histogram(p2[:,:,0],256,[0,256]),np.histogram(p2[:,:,1],256,[0,256]),np.histogram(p2[:,:,2],256,[0,256])
    h1 = np.concatenate((h10[0],h11[0],h12[0]))
    h2 = np.concatenate((h20[0],h21[0],h22[0]))
    return -np.mean((h1-h2)**2)

def iou(bb_test,bb_gt):
    ### bb_*: [x1,y1,x2,y2]
    xx1 = np.maximum(bb_test[0], bb_gt[0])
    yy1 = np.maximum(bb_test[1], bb_gt[1])
    xx2 = np.minimum(bb_test[2], bb_gt[2])
    yy2 = np.minimum(bb_test[3], bb_gt[3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w*h
    o = wh / ((bb_test[2]-bb_test[0])*(bb_test[3]-bb_test[1])
        + (bb_gt[2]-bb_gt[0])*(bb_gt[3]-bb_gt[1]) - wh)
    return o


def data_associate(score_mat):
    detnum,targetnum = score_mat.shape
    matched_indices = linear_assignment(-score_mat)
    #print score_mat
    unm_dets = []
    for d in range(detnum):
        if(d not in matched_indices[:,0]):
            unm_dets.append(d)
    unm_targets = []
    for t in range(targetnum):
        if(t not in matched_indices[:,1]):
            unm_targets.append(t)

    matches = []
    match_thres = -10000
    for m in matched_indices:
        if(score_mat[m[0],m[1]])<match_thres:
            unm_dets.append(m[0])
            unm_targets.append(m[1])
        else: matches.append(m)

    return matches, unm_dets, unm_targets


class Tracker(object):
    def __init__(self):
        self.targetlst = []
        self.frameidx = 0

    def getScoreMatrix(self, img, dets):
        score_mat = np.zeros((len(dets),len(self.targetlst)))
        for i, det in enumerate(dets):
            for j, target in enumerate(self.targetlst):
                x1,y1,x2,y2,det_score = det
                score_mat[i,j] = target.match(det, img[y1:y2,x1:x2])
        #print score_mat
        return score_mat

    def checkOverlap(self, bb):
        for target in self.targetlst:
            if iou(bb, target.motion.get_state())>0.05 and target.state=='live':
                return True
        return False
        
    def initFromImgNDets(self, img, dets):
        init_thres = -100000
        new_targets = []
        for det in dets:
            x1,y1,x2,y2,det_score = det
            if self.checkOverlap([x1,y1,x2,y2]): continue
            if det_score>init_thres:
                self.targetidx+=1
                target = TrackedTarget(self.targetidx,[x1,y1,x2,y2],patch=img[y1:y2,x1:x2])
                #new_targets.append(target)
                self.targetlst.append(target)
        #self.targetlst+=new_targets

    def processFirstImgNDets(self, img, dets):
        self.frameidx+=1
        self.targetidx = 0
        self.initFromImgNDets(img,dets)

    def processNextImg(self, img):
        pass

    def processNextImgNDets(self, img, dets):
        self.frameidx+=1
        score_mat = self.getScoreMatrix(img,dets)
        matches,unm_det_idxs,unm_target_idxs = data_associate(score_mat)

        for m in matches:
            x1,y1,x2,y2,det_score = dets[m[0]]
            self.targetlst[m[1]].update([x1,y1,x2,y2], img[y1:y2,x1:x2])

        unm_dets = [dets[i] for i in unm_det_idxs]
        self.initFromImgNDets(img, unm_dets)
        
        for t in unm_target_idxs:
            self.targetlst[t].report_miss()

        self.targetlst = [target for target in self.targetlst if target.state!='gone']
        

    def output(self, out):
        for target in self.targetlst:
            if target.state=='live' and target.life>2:
            #if target.life>0:
                s = target.output()
                out.write(str(self.frameidx)+','+s+'\n')

    def plot(self, img, codelst):
        for target in self.targetlst:
            if target.state=='live':
            #if target.life>0 and target.state=='live':
            #if target.state=='live' and target.life>2:
                #print target.idx
                x1,y1,x2,y2 = target.motion.get_state()
                x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
                cv2.rectangle(img,(x1,y1),(x2,y2), codelst[target.idx%len(codelst)], 2)
        cv2.imshow('tracking plot', img)


class TrackedTarget(object):
    def __init__(self, idx, bb, patch=None):
        self.appearences = []
        if patch is not None: self.appearences.append(patch)
        self.idx, self.life, self.miss = idx, 0, 0
        self.state = 'live'
        self.motion = LinearMotionModel(bb)


    def match(self, det, patch=None):
        bb = det[:4]
        ### motion model
        last_bb = self.motion.get_state()
        #print iou(last_bb,bb)
        if iou(last_bb,bb)<0.3: return -100000
        ### appearence model
        score_a = color_sim(self.appearences[-1],patch)
        
        #return score_a
        return iou(last_bb,bb)

    def update(self, det, patch=None):
        self.life+=1
        self.miss = 0

        if patch is not None: self.appearences = [patch]
        bb = det[:4]
        self.state = 'live'
        self.motion.update(bb)
        ### TODO memory network

    def report_miss(self):
        self.miss+=1
        self.state = 'miss'
        self.motion.predict()
        if self.miss>1: self.life = 0
        if self.miss>2: self.state = 'gone'

    def output(self):
        x1,y1,x2,y2 = self.motion.get_state()
        s = str(self.idx)+','+str(int(x1))+','+str(int(y1))+','+str(int(x2-x1))+','+str(int(y2-y1))+',1,-1,-1,-1'
        return s


def bb2z(bb):
    w = bb[2]-bb[0]
    h = bb[3]-bb[1]
    x = bb[0]+w/2.
    y = bb[1]+h/2.
    s = w*h
    r = w/float(h)
    return [x,y,s,r]
    
def z2bb(z):
    w = np.sqrt(z[2]*z[3])
    h = z[2]/w
    return [z[0]-w/2.,z[1]-h/2.,z[0]+w/2.,z[1]+h/2.]

class LinearMotionModel(object):
    def __init__(self,bb):
        self.bb = bb
        self.z_his = [bb2z(bb)]
        self.len_thres = 5

    def update(self,bb):
        self.bb = bb
        self.z_his.append(bb2z(bb))
        if len(self.z_his)>self.len_thres: self.z_his.pop(0)

    def predict(self):
        self.bb = None
        his = np.array(self.z_his)
        if his.shape[0]==1:
            self.z_his*=2
        else:
            t = np.array(range(his.shape[0]))
            t_mean, t_var = np.mean(t), np.var(t)
            x_mean, y_mean = np.mean(his[:,0]), np.mean(his[:,1])
            cov_xt = np.cov(his[:,0],y=t)[0,1]
            cov_yt = np.cov(his[:,1],y=t)[0,1]
            ax, ay = cov_xt/t_var, cov_yt/t_var
            bx, by = x_mean-ax*t_mean, y_mean-ay*t_mean

            x,y = ax*his.shape[0]+bx, ay*his.shape[0]+by
            s,r = np.mean(his[:,2]), np.mean(his[:,3])

            self.z_his.append([x,y,s,r])
            if len(self.z_his)>self.len_thres: self.z_his.pop(0)

    def get_state(self):
        if self.bb is None: return z2bb(self.z_his[-1])
        return self.bb
