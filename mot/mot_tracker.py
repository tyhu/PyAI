"""
Multiple Object Tracking Framework
"""
import sys
import copy
import numpy as np
from random import shuffle
from sklearn.utils.linear_assignment_ import linear_assignment
from motion_models import *
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
    match_thres = 0.3
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
                x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
                #score_mat[i,j] = target.match(det, img[y1:y2,x1:x2])
                score_mat[i,j] = target.match(det, None)
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
            x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
            #if self.checkOverlap([x1,y1,x2,y2]): continue
            if det_score>init_thres:
                self.targetidx+=1
                #target = TrackedTarget(self.targetidx,[x1,y1,x2,y2],patch=img[y1:y2,x1:x2])
                target = TrackedTarget(self.targetidx,[x1,y1,x2,y2],patch=None)
                #new_targets.append(target)
                self.targetlst.append(target)
        #self.targetlst+=new_targets

    def processFirstImgNDets(self, img, dets):
        self.frameidx+=1
        self.targetidx = 0
        self.initFromImgNDets(img,dets)

    def processNextImg(self, img):
        pass

    def processNextImgNDets(self, img, dets, record=False):
        self.frameidx+=1
        score_mat = self.getScoreMatrix(img,dets)
        matches,unm_det_idxs,unm_target_idxs = data_associate(score_mat)

        for m in matches:
            x1,y1,x2,y2,det_score = dets[m[0]]
            x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
            #self.targetlst[m[1]].update([x1,y1,x2,y2], img[y1:y2,x1:x2])
            self.targetlst[m[1]].update([x1,y1,x2,y2], None)

        unm_dets = [dets[i] for i in unm_det_idxs]
        self.initFromImgNDets(img, unm_dets)
        
        for t in unm_target_idxs:
            self.targetlst[t].report_miss()

        self.targetlst = [target for target in self.targetlst if target.state!='gone']
        

    def output(self, out):
        for target in self.targetlst:
            if (target.state=='live' and target.life>2) or self.frameidx<4:
            #if (target.state=='live' and target.life>2):
            #if target.state=='live' and target.life>3:
            #if target.life>1:
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
        #self.motion = LinearMotionModel(bb)
        self.motion = KFMotionModel(bb)
        self.gtidx = -1

    def match_feat(self, det, patch=None):
        bb = det[:4]
        #last_bb = self.motion.get_state()
        #last_bb = self.motion.get_predicted_state()
        last_bb = self.motion.predict()
        iouv = iou(last_bb,bb)
        #score_a = color_sim(self.appearences[-1],patch)
        last_bb2 = self.motion.get_state()
        iouv2 = iou(last_bb2,bb)
        return iouv, iouv2
        #return iouv, -10/np.minimum(score_a,-20)

    def match(self, det, patch=None):
        bb = det[:4]
        ### motion model
        #last_bb = self.motion.get_state()
        last_bb = self.motion.predict()
        #print iou(last_bb,bb)
        #if iou(last_bb,bb)<0.01: return -1
        ### appearence model
        #score_a = color_sim(self.appearences[-1],patch)
        
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

    def report_miss(self,force=False):
        self.miss+=1
        self.state = 'miss'
        self.motion.predict()
        if self.miss>1: self.life = 0
        #if self.miss>2 and not force: self.state = 'gone'
        if self.miss>2 and not force: self.state = 'gone'

    def output(self):
        x1,y1,x2,y2 = self.motion.get_state()
        s = str(self.idx)+','+str(int(x1))+','+str(int(y1))+','+str(int(x2-x1))+','+str(int(y2-y1))+',1,-1,-1,-1'
        return s


def validDecision(mat, dnum, tnum):
    ### detection
    for i in range(dnum):
        if np.sum(mat[i,:])!=1: return False
    for j in range(tnum):
        if np.sum(mat[:,j])!=1: return False

    if np.sum(mat[dnum:,tnum:])!=0: return False

    for i in range(dnum):
        for j in range(dnum):
            if i==j: continue
            if mat[i,tnum+j]==1 or mat[i,tnum+dnum+j]==1: return False

    for i in range(tnum):
        for j in range(tnum):
            if i==j: continue
            if mat[dnum+i,j]==1 or mat[dnum+tnum+i,j]==1: return False

    return True

class StructMDPTracker(Tracker):
    def __init__(self):
        self.featidxs = {'match':0,'drop':4,'initial':8,'miss':11,'delete':13}
        self.fsize = 15
        #self.w = np.ones(self.fsize)
        self.w = np.array([1000,0,1,0,0,20,0.01,0.01,0.01,0.05,0.05,0.05,0.2,0.2,0.1,0.1])
        self.reset()
        self.def_feat2()

    def checkOverlap(self,bb):
        ioulst = [iou(bb, target.motion.get_state()) for target in self.targetlst]
        ioulst+=[0.0]
        return np.max(ioulst)*5

    def def_feat2(self):
        self.featidxs = {'match':0,'drop':1,'initial':2,'miss':3,'delete':4}
        self.fsize = 5

    #def genStateMap2(self, img, dets):
    def generateStateMap(self, img, dets):
        dnum, tnum = len(dets), len(self.targetlst)
        state_map = np.zeros((dnum+2*tnum,tnum+2*dnum,self.fsize))
        ### drop/initialize detection
        for i,det in enumerate(dets):
            drop_idx, init_idx = self.featidxs['drop'], self.featidxs['initial']
            x1,y1,x2,y2,det_score = det
            x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
            overlap = self.checkOverlap([x1,y1,x2,y2])
            state_map[i,tnum+i,drop_idx:drop_idx+1] = np.array([overlap])
            state_map[i,tnum+dnum+i,init_idx:init_idx+1] = np.array([1])

        ### report miss/delete target
        miss_idx, delete_idx = self.featidxs['miss'], self.featidxs['delete']
        for i,target in enumerate(self.targetlst):
            feat = np.array([1])
            state_map[dnum+i,i,miss_idx:miss_idx+1] = feat
            state_map[dnum+tnum+i,i,delete_idx:delete_idx+1] = feat
        
        ### match det and target
        match_idx = self.featidxs['match']
        for i,det in enumerate(dets):
            for j,target in enumerate(self.targetlst):
                x1,y1,x2,y2,det_score = det
                x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
                iou, csim = target.match_feat(det, None)
                state_map[i,j,match_idx:match_idx+1] = np.array([iou])
        return state_map


    def genStateMap2(self, img, dets):
    #def generateStateMap(self, img, dets):
        dnum,tnum = len(dets), len(self.targetlst)
        state_map = np.zeros((dnum+2*tnum,tnum+2*dnum,self.fsize))
        ## get state-action feature map
        ### drop/initialize detection
        for i,det in enumerate(dets):
            drop_idx, init_idx = self.featidxs['drop'], self.featidxs['initial']
            x1,y1,x2,y2,det_score = det
            x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
            size = (x2-x1)*(y2-y1)/1000
            overlap = self.checkOverlap([x1,y1,x2,y2])
            state_map[i,tnum+i,drop_idx:drop_idx+4] = np.array([overlap,size,det_score,1])
            state_map[i,tnum+dnum+i,init_idx:init_idx+3] = np.array([size,det_score,1])
        ### report miss/delete target
        miss_idx, delete_idx = self.featidxs['miss'], self.featidxs['delete']
        for i,target in enumerate(self.targetlst):
            feat = np.array([target.miss/10,1])
            state_map[dnum+i,i,miss_idx:miss_idx+2] = feat
            state_map[dnum+tnum+i,i,delete_idx:delete_idx+2] = feat
        ### match det and target
        match_idx = self.featidxs['match']
        for i,det in enumerate(dets):
            for j,target in enumerate(self.targetlst):
                x1,y1,x2,y2,det_score = det
                x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
                #iou, csim = target.match_feat(det, img[y1:y2,x1:x2])
                iou, csim = target.match_feat(det, None)

                state_map[i,j,match_idx:match_idx+4] = np.array([iou,csim,target.miss,1])
        return state_map


    def get_deci_map(self, state_map, w, dnum, tnum):
        score_mat = state_map.dot(w)
        match_idxs = linear_assignment(-score_mat)
        deci_map = np.zeros_like(score_mat)
        for m in match_idxs:
            if m[0]<dnum or m[1]<tnum:
                deci_map[m[0],m[1]] = 1
        return match_idxs, deci_map


    def apply_deci_map(self, deci_map, dets, img):
        dnum,tnum = len(dets), len(self.targetlst)
        for i in range(dnum):
            x1,y1,x2,y2,det_score = dets[i]
            x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
            j = deci_map[i,:].tolist().index(1)
            if j<tnum:
                #print 'match!'
                #self.targetlst[j].update([x1,y1,x2,y2], img[y1:y2,x1:x2])
                self.targetlst[j].update([x1,y1,x2,y2], None)
            elif j>=tnum+dnum:
                self.targetidx+=1
                #target = TrackedTarget(self.targetidx,[x1,y1,x2,y2],patch=img[y1:y2,x1:x2])
                target = TrackedTarget(self.targetidx,[x1,y1,x2,y2],patch=None)
                self.targetlst.append(target)
                #print 'initialize!'
            else:
                #print 'drop!'
                continue

        for j in range(tnum):
            i = deci_map[:,j].tolist().index(1)
            if i>=dnum and i<dnum+tnum:   #report miss
                #print 'miss!'
                self.targetlst[j].report_miss(force=False)
            elif i>=dnum+tnum:
                self.targetlst[j].state = 'gone'
                #print 'delete!'
        self.targetlst = [target for target in self.targetlst if target.state!='gone']


    def processNextImgNDets(self, img, dets):
        self.frameidx+=1
        dnum,tnum = len(dets), len(self.targetlst)
        state_map = self.generateStateMap(img, dets)

        mach_idxs, deci_map = self.get_deci_map(state_map, self.w, dnum, tnum)
        self.apply_deci_map(deci_map, dets, img)
        #esti_reward = np.sum(his.dot(self.w))
        #self.state_action_his.append(his.tolist())
        #self.esti_reward_lst.append(esti_reward)
        #print validDecision()
        #print self.targetidx
        #raw_input()

    def processFirstImgNDets(self, img, dets):
        self.targetidx = 0
        self.processNextImgNDets(img,dets)

    def initFirstImg(self, img, dets):
        self.targetidx = 0
        for det in dets:
            x1,y1,x2,y2,det_score = det
            x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
            self.targetidx+=1
            #target = TrackedTarget(self.targetidx,[x1,y1,x2,y2],patch=img[y1:y2,x1:x2])
            target = TrackedTarget(self.targetidx,[x1,y1,x2,y2],patch=None)
            self.targetlst.append(target)

    def reset(self):
        self.targetlst = []
        self.frameidx = 0
        self.state_action_his = []
        self.esti_reward_lst = []

        ### for training
        self.state_his = []
        self.action_his = []
        self.feats, self.labs = [],[]
        self.dlst = []

    def extract_gt_decision(self,gt,dets,img,appl=True,thres=0.05):
        dnum, tnum = len(dets), len(self.targetlst)
        deci_map = np.zeros((dnum+2*tnum,tnum+2*dnum))
        gtnum = len(gt)
        det_gt_mat = np.empty((dnum,gtnum))
        for i in range(dnum):
            for j in range(gtnum):
                det_gt_mat[i,j] = iou(dets[i],gt[j][2:6])
        det_gt_matches = linear_assignment(-det_gt_mat)
        det_gt_matches = [m for m in det_gt_matches if det_gt_mat[m[0],m[1]]>thres]
        gt_taridx_lst = gt[:,1].tolist()
        gt_idxs, det_idxs = range(gtnum), range(dnum)
        ### match or initialize
        curlst = [t.idx for t in self.targetlst]
        for m in det_gt_matches:
            #x1,y1,x2,y2 = gt[m[1],2:6]
            x1,y1,x2,y2 = dets[m[0],:4]
            x1,y1,x2,y2 = int(x1), int(y1),int(x2),int(y2)
            if gt[m[1],1] in curlst:  ### match
                tidx = curlst.index(gt[m[1],1])
                deci_map[m[0],tidx] = 1
                if appl: self.targetlst[tidx].update([x1,y1,x2,y2], img[y1:y2,x1:x2])
            else:   ### initialize
                targetidx = gt[m[1],1]
                target = TrackedTarget(targetidx, [x1,y1,x2,y2], patch=img[y1:y2,x1:x2])
                if appl: self.targetlst.append(target)
                deci_map[m[0],tnum+dnum+m[0]] = 1
        matched_det_lst = [m[0] for m in det_gt_matches]
        for i in range(dnum):
            if i not in matched_det_lst: ### drop
                deci_map[i,tnum+i] = 1

        matched_gt_lst = [m[1] for m in det_gt_matches]
        for j,gtidx in enumerate(gt_taridx_lst):
            if j not in matched_gt_lst: ### miss
                if gtidx in curlst:
                    i = curlst.index(gtidx)
                    deci_map[dnum+i,i] = 1
                #if not in curlst: forget it, initialize it next frame

        for i,tidx in enumerate(curlst):
            if tidx not in gt_taridx_lst:
                self.targetlst[i].state = 'gone'
                deci_map[dnum+tnum+i,i] = 1
        if appl: self.targetlst = [target for target in self.targetlst if target.state!='gone']
        return deci_map


    ### extract ground truth history
    def processNextGT(self, img, gt, dets):
        self.frameidx+=1
        state_map = self.generateStateMap(img, dets)
        deci_map = self.extract_gt_decision(gt,dets,img,appl=False)
        fmaplst = self.disturb_decision(deci_map, gt, dets, img)
        shuffle(fmaplst)
        deci_map = self.extract_gt_decision(gt,dets,img,appl=True)
        #if len(fmaplst)>0:
        #    f1 = self.struct_feat(state_map,deci_map)
        #    f2 = self.struct_feat(state_map,fmaplst[0])
        #    f3 = self.struct_feat(state_map,fmaplst[-1])
        #    self.feats+=[f1-f2,f1-f3]
        fg = self.struct_feat(state_map,deci_map)
        for fmap in fmaplst:
            f1 = self.struct_feat(state_map,fmap)
            self.feats.append(fg-f1)
            
        self.state_his.append(state_map)
        self.action_his.append(deci_map)
        #self.labs+=[1,-1]
        #if self.frameidx==1:
        #    self.feats+=[f1-f2,f3-f1]*10
        #    self.labs+=[1,-1]*10

    def disturb_decision(self, deci_map, gt, dets, img):
        dnum, gtnum, tnum = len(dets), len(gt), len(self.targetlst)
        fmaplst = []
        snum = 2

        if gtnum>0:
            ssnum = np.minimum(snum,gtnum)
            idxs = np.random.choice(gtnum,ssnum,replace=False).tolist()
            #dis_gt = copy.deepcopy(gt)
            for i in idxs:
                dis_gt = np.delete(gt,i,0)
                fmaplst.append(self.extract_gt_decision(dis_gt,dets,img,appl=False))
                
        
        ### find drop, initial
        droplst, initiallst = [],[]
        for i in range(dnum):
            de = deci_map[i,:].tolist().index(1)
            if de>tnum and de<tnum+dnum: droplst.append(i)
            elif de>tnum+dnum: initiallst.append(i)
        if len(droplst)+len(initiallst)>0:
            ssnum = np.minimum(snum,len(droplst)+len(initiallst))
            idxs = np.random.choice(len(droplst)+len(initiallst),ssnum,replace=False)
            for i in idxs:
                f_deci_map2 = copy.deepcopy(deci_map)
                i = np.random.choice(droplst+initiallst)
                f_deci_map2[i,:]*=0
                if i in droplst: f_deci_map2[i,dnum+tnum+i] = 1
                else: f_deci_map2[i,tnum+i] = 1
                fmaplst.append(f_deci_map2)

        ### find delete and miss
        misslst, deletelst = [],[]
        for i in range(tnum):
            de = deci_map[:,i].tolist().index(1)
            if de>dnum and de<dnum+tnum: misslst.append(i)
            elif de>dnum+tnum: deletelst.append(i)
        if len(misslst)+len(deletelst)>0:
            ssnum = np.minimum(snum,len(misslst)+len(deletelst))
            idxs = np.random.choice(len(misslst)+len(deletelst),ssnum,replace=False)
            for i in idxs:
                f_deci_map3 = copy.deepcopy(deci_map)
                i = np.random.choice(misslst+deletelst)
                f_deci_map3[:,i]*=0
                if i in misslst: f_deci_map3[dnum+tnum+i,i] = 1
                else: f_deci_map3[dnum+i,i] = 1
                fmaplst.append(f_deci_map3)

        return fmaplst


    def struct_feat(self, state_map, deci_map):
        x,y,featnum = state_map.shape
        feat = np.zeros((featnum,)).astype('float')
        for i in range(x):
            for j in range(y):
                if deci_map[i,j] == 1: feat+=state_map[i,j,:]
        return feat
        
    def process_gt_collect_data(self, img, gt, dets, w1, w2=None):
        self.frameidx+=1
        dnum, tnum = len(dets), len(self.targetlst)
        state_map = self.generateStateMap(img, dets)
        deci_map = self.extract_gt_decision(gt,dets,img,appl=False)
        feat = self.struct_feat(state_map, deci_map)

        _, deci_map1 = self.get_deci_map(state_map,w1,dnum,tnum)
        feat1 = self.struct_feat(state_map, deci_map1)
        d1 = np.sum(np.abs(deci_map-deci_map1))

        if w2 is not None:
            _, deci_map2 = self.get_deci_map(state_map,w2,dnum,tnum)
            feat2 = self.struct_feat(state_map, deci_map2)
            d2 = np.sum(np.abs(deci_map-deci_map2))

        if d1>0: self.feats.append(feat-feat1)
        self.totald+=d1
        #print d, deci_map.shape
        #if d>0: self.feats.append(feat-feat2)
        #if d1>d2: self.feats.append(feat2-feat1)
        #elif d1<d2: self.feats.append(feat1-feat2)
        #if np.sum(np.abs(deci_map-deci_map1))!=0: self.feats.append(feat-feat1)
        #if np.sum(np.abs(deci_map-deci_map2))!=0: self.feats.append(feat-feat2)
        
        deci_map = self.extract_gt_decision(gt,dets,img,appl=True)

    def process_img_collect_data(self,img, gt, gt_1, dets):
        self.frameidx+=1
        dnum,tnum = len(dets), len(self.targetlst)
        state_map = self.generateStateMap(img, dets)

        match_idxs, deci_map = self.get_deci_map(state_map, self.w, dnum, tnum)

        deci_map_gt = self.deci_approach_gt(dets, gt, gt_1)
        d = np.sum(np.abs(deci_map-deci_map_gt))
        self.apply_deci_map(deci_map, dets, img)

        feat = self.struct_feat(state_map, deci_map)
        feat_gt = self.struct_feat(state_map, deci_map_gt)
        #if d>2 and d<6: self.feats.append(feat_gt-feat)
        if d>0:
            self.feats.append(feat_gt-feat)
            self.dlst.append(d)


    """
    def interpolate_decision(self,deci_map, deci_map_gt, dnum, tnum,p=0.5):
        deci_map_copy = np.array(deci_map, copy=True)
        for d in range(dnum):
            if not np.array_equal(deci_map[d,:], deci_map_gt[d,:]):
                t1 = deci_map[d,:].tolist().index(1)
                t2 = deci_map_gt[d,:].tolist().index(1)
                if np.random.rand(1)<p:
                    deci_map_copy[d,:] = deci_map_gt[d,:]

        for t in range(tnum):
            if not np.array_equal(deci_map[:,t], deci_map_gt[:,t]):
    """

    ### make a decision so that the current state approaches ground truth
    def deci_approach_gt(self, dets, gt, gt_1=[]):
        dnum, tnum = len(dets), len(self.targetlst)
        deci_map = np.zeros((dnum+2*tnum,tnum+2*dnum))

        ### align current targets to gt_1
        gtnum_1 = len(gt_1)
        if gtnum_1==0: target_gidxs = []
        else:
            mat_1 = np.zeros((tnum,gtnum_1))
            for i in range(tnum):
                bb = self.targetlst[i].motion.get_state()
                for j in range(gtnum_1):
                    mat_1[i,j] = iou(bb,gt_1[j][2:6])
            match_1 = linear_assignment(-mat_1)
            for m in match_1:
                i,j = m[0],m[1]
                if mat_1[i,j]>0.1: self.targetlst[i].gtidx = gt_1[j][1]
            target_gidxs = [self.targetlst[i].gtidx for i in range(len(self.targetlst))]
               

        ### align dets to gt
        gtnum = len(gt)
        mat_det_gt = np.zeros((dnum,gtnum))
        for i in range(dnum):
            for j in range(gtnum):
                mat_det_gt[i,j] = iou(dets[i],gt[j][2:6])
        match_det_gt = linear_assignment(-mat_det_gt)
        gtlst = gt[:,1].tolist()
        
        ### infer decisions
        gt_match_lst = [m[1] for m in match_det_gt]
        for i in range(tnum):
            target = self.targetlst[i]
            if target.gtidx not in gtlst:
                #print 'gt: delete'
                deci_map[dnum+tnum+i,i] = 1
            else:
                j = gtlst.index(target.gtidx)
                if j in gt_match_lst:
                    l = gt_match_lst.index(j)
                    k = match_det_gt[l][0]
                    #print 'gt: match'
                    deci_map[k,i] = 1
                else:
                    #print 'gt: miss'
                    deci_map[dnum+i,i] = 1

        det_match_lst = [m[0] for m in match_det_gt]
        for i in range(dnum):
            if i not in det_match_lst:
                #print 'gt: drop'
                deci_map[i, tnum+i] = 1
            else:
                l = det_match_lst.index(i)
                k = match_det_gt[l][1]
                gidx = gt[k,1]
                if gidx not in target_gidxs:
                    #print 'gt: initialize'
                    deci_map[i, tnum+dnum+i] = 1

        return deci_map
