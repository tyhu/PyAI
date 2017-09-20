import sys
sys.path.insert(0,"/home/tingyaoh/.local/lib/python2.7/site-packages/")
sys.path.insert(0,"./eval/")
import motio
import util_moteval
import mot_metrics
import numpy as np
from myconfig import *
from util_track import *
from mot_tracker import *
from sklearn.svm import LinearSVC
import pickle
from myopt import *

def mot_evaluate_list(gtfnlst, outfnlst):
    for gtfn, outfn in zip(gtfnlst,outfnlst):
        df_gt = motio.loadtxt(gtfn)
        df_test = motio.loadtxt(outfn)

def mot_evaluate(gtfn, outfn):
    df_gt = motio.loadtxt(gtfn)
    df_test = motio.loadtxt(outfn)
    acc = util_moteval.compare_to_groundtruth(df_gt, df_test, 'iou', distth=0.5)
    mh = mot_metrics.create()
    res = mh.compute(acc, metrics=mot_metrics.motchallenge_metrics)
    res = res.to_dict()
    return res
    

def train(tracker, detfn, imgdir, gtfn):
    outfn = 'train_tmp.txt'
    #tracker.w = np.array([1000,0,1,0,0,20,0.01,0.01,0.01,0.05,0.05,0.05,0.1,0.1,0.2,0.2])/50
    tracker.w = np.array([500,0,1,0,0,20,0.01,0.01,0.01,0.05,0.05,0.05,0.1,0.1,0.2,0.2])/2000
    for it in range(100):
        tracker.reset()
        outfile = file(outfn,'w')
        for frid,img,patchlst,dets in yieldMOTImgDet(detfn,imgdir):
            if frid==1:
                tracker.processFirstImgNDets(img,dets)
            else:
                tracker.processNextImgNDets(img,dets)
            tracker.output(outfile)
        outfile.close()
        res = mot_evaluate(gtfn, outfn)
        reward = res['mota'][0]*100
        print 'reward: ',reward
        #print 'estimated reward:', np.m
        FAQL(tracker, reward)

def training():
    for seq in train_seq_list:
        print 'Training ',seq
        tracker = StructMDPTracker()
        detfn = datadir+'/'+seq+'/det/det.txt'
        #detfn = '/home/tingyaoh/github/sort/data/'+seq+'/det.txt'
        imdir = datadir+'/'+seq+'/img1/'
        gtfn = datadir+'/'+seq+'/gt/gt.txt'
        train(tracker,detfn,imdir,gtfn)
        break
    

def test():

    codelst = colorcodes()
    display = False
    for seq in train_seq_list:
        print 'Tracking ',seq
        tracker = Tracker()
        #tracker = StructMDPTracker()
        #detfn = datadir+'/'+seq+'/det/det.txt'
        detfn = '/home/tingyaoh/github/sort/data/'+seq+'/det.txt'
        imdir = datadir+'/'+seq+'/img1/'
        gtfn = datadir+'/'+seq+'/gt/gt.txt'
        outfn = 'output/'+seq+'.txt'
        #plot_detection(gtfn,imdir)
    
        if not display: outfile = file(outfn,'w')
        for frid,img,patchlst,dets in yieldMOTImgDet(detfn,imdir):
            if frid==1:
                tracker.processFirstImgNDets(img,dets)
            else:
                tracker.processNextImgNDets(img,dets)
            if not display: tracker.output(outfile)
            if display:
                tracker.plot(img,codelst)
                cv2.waitKey(-1)
        if not display: outfile.close()
        res = mot_evaluate(gtfn, outfn)
        print res['mota'][0]


def test_w(w,seq_list):
    for i,seq in enumerate(seq_list):
        print 'Tracking ',seq
        tracker = StructMDPTracker()
        tracker.w = np.array(w)
        #detfn = datadir+'/'+seq+'/det/det.txt'
        detfn = '/home/tingyaoh/github/sort/data/'+seq+'/det.txt'
        imdir = datadir+'/'+seq+'/img1/'
        gtfn = datadir+'/'+seq+'/gt/gt.txt'
        outfn = 'output/'+seq+'.txt'
        #plot_detection(gtfn,imdir)
    
        outfile = file(outfn,'w')
        for frid,img,patchlst,dets in yieldMOTImgDet(detfn,imdir):
            if frid==1:
                tracker.processFirstImgNDets(img,dets)
            else:
                tracker.processNextImgNDets(img,dets)
            tracker.output(outfile)
        outfile.close()
        res = mot_evaluate(gtfn, outfn)
        print res['mota'][0]
        #if i==1:
        #    break


def test_collect(w):
    w1 = w
    w2 = np.array([1000,0,1,0,0,20,0.01,0.01,0.01,0.05,0.05,0.05,0.1,0.1,0.2,0.2])
    featlst = []
    d = 0
    for i,seq in enumerate(train_seq_list):
        print 'Tracking ',seq
        tracker = StructMDPTracker()
        detfn = '/home/tingyaoh/github/sort/data/'+seq+'/det.txt'
        imdir = datadir+'/'+seq+'/img1/'
        gtfn = datadir+'/'+seq+'/gt/gt.txt'

        for frid,patchlst,img,dets,gt in yieldMOTDetNGt(detfn,imdir,gtfn):
            tracker.process_gt_collect_data(img, gt, dets, w1, w2)
        d+=tracker.totald
        featlst+=tracker.feats
        if i==1: break
    feats = np.array(featlst)
    print 'estimate distance:', d
    #print feats.shape
    #pickle.dump(feats,open('feats2.pkl','wb'))
    return feats
    
def test_gt2(w):
    featlst = []
    for i,seq in enumerate(train_seq_list):
        print 'Tracking ',seq
        tracker = StructMDPTracker()
        tracker.w = w
        #detfn = datadir+'/'+seq+'/det/det.txt'
        detfn = '/home/tingyaoh/github/sort/data/'+seq+'/det.txt'
        imdir = datadir+'/'+seq+'/img1/'
        gtfn = datadir+'/'+seq+'/gt/gt.txt'
        outfn = 'output/'+seq+'.txt'
        for frid,patchlst,img,dets,gt in yieldMOTDetNGt(detfn,imdir,gtfn):
            if frid==1:
                tracker.processFirstImgNDets(img,dets)
            else:
                tracker.process_img_collect_data(img, gt, gt_1, dets)
            gt_1 = gt
        featlst+=tracker.feats
        if i==1: break
    feats = np.array(featlst)
    print feats.shape
    return feats
    

def test_gt():
    display = False
    featlst = []
    lablst = []
    for i,seq in enumerate(train_seq_list):
        print 'Tracking ',seq
        tracker = StructMDPTracker()
        #detfn = datadir+'/'+seq+'/det/det.txt'
        detfn = '/home/tingyaoh/github/sort/data/'+seq+'/det.txt'
        imdir = datadir+'/'+seq+'/img1/'
        gtfn = datadir+'/'+seq+'/gt/gt.txt'
        outfn = 'output/'+seq+'.txt'
        #plot_detection(gtfn,imdir)
    
        if not display: outfile = file(outfn,'w')
        for frid,patchlst,img,dets,gt in yieldMOTDetNGt(detfn,imdir,gtfn):
            tracker.processNextGT(img, gt, dets)
        featlst+=tracker.feats
        #res = mot_evaluate(gtfn, outfn)
        #print res['mota'][0]
        if i==1: break
    feats = np.array(featlst)
    print feats.shape
    pickle.dump(feats,open('feats_gt.pkl','wb'))
    #return w

def iterative_train(it=3):
    feats = pickle.load(open('feats.pkl','rb'))
    w = solve_tracking_qp(feats,0.001)
    wlst = [w]
    for i in range(it):
        #feats2 = test_collect(wlst[-1])
        feats2 = test_gt2(wlst[-1])
        feats = np.concatenate((feats,feats2),axis=0)
        #feats = uniform_sample(feats)
        #w = reduced_GD_solve(feats,0.01,init_w=w,lr=0.001,iternum=20000)
        print feats.shape
        w = solve_tracking_qp(feats,0.001)
        wlst.append(w)
        #test_w(w,train_seq_list)
    pickle.dump(feats, open('feats_accu.pkl','wb'))
    return wlst

def uniform_sample(feats):
    datanum, _ = feats.shape
    if datanum>5000:
        idxs = np.random.choice(datanum,5000,replace=False)
        feats = feats[idxs,:]
    return feats

if __name__=='__main__':
    #tesat()
    #training()
    #test_gt()
    #test_collect()
    
    #wlst = iterative_train(it=2)
    #feats = pickle.load(open('feats_accu.pkl','rb'))
    feats = pickle.load(open('feats_gt.pkl','rb'))
    #feats = pickle.load(open('feats.pkl','rb'))
    #datanum,_ = feats.shape
    #w = np.array([1000,0,1,0,0,20,0.01,0.01,0.01,0.05,0.05,0.05,0.2,0.2,0.1,0.1])/10000
    #print objective_tracking(w,feats,0.01)
    #print feats.shape
    #test_w(w,train_seq_list)
    #w = solve_tracking_qp(feats, 0.001)

    
    w = solve_tracking_qp(feats, 0.001)
    w = np.array(w)
    print 'testing..'
    print zero_one_loss(feats,w)
    test_w(w,train_seq_list)
    test_w(w,valid_seq_list)
    pickle.dump(feats,open('feats_curr.pkl','wb'))
    

