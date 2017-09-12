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


def test_collect():
    w1 = np.array([0.445791961848354, 0.0, 0.49337954015946095, 0.4087116870708428, 0.6307439693110243, 0.044208188310918356, 0.0006677631578946937, 0.2874547028084139, 0.1747404743850009, 0.0009524291497975709, 0.3238557138652923, 0.1347388663967141, 0.7642764977391971, 0.5797327735063608, 0.3969737776023806, 0.5692823020748424]) 
    w2 = np.array([1000,0,1,0,0,20,0.01,0.01,0.01,0.05,0.05,0.05,0.1,0.1,0.2,0.2])
    featlst = []
    for i,seq in enumerate(train_seq_list):
        print 'Tracking ',seq
        tracker = StructMDPTracker()
        detfn = '/home/tingyaoh/github/sort/data/'+seq+'/det.txt'
        imdir = datadir+'/'+seq+'/img1/'
        gtfn = datadir+'/'+seq+'/gt/gt.txt'

        for frid,patchlst,img,dets,gt in yieldMOTDetNGt(detfn,imdir,gtfn):
            tracker.process_gt_collect_data(img, gt, dets, w1, w2)
        featlst+=tracker.feats
    feats = np.array(featlst)
    print feats.shape
    pickle.dump(feats,open('feats2.pkl','wb'))
    

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
        if i>2: break
    feats = np.array(featlst)
    print feats.shape
    pickle.dump(feats,open('feats.pkl','wb'))
    #return w

if __name__=='__main__':
    #test()
    #training()
    #test_gt()
    #test_collect()
    
    
    feats = pickle.load(open('feats.pkl','rb'))
    #for feat in feats.tolist(): print feat
    
    #feats[:,1:4] = 0
    w = reduced_GD_solve(feats,0.01,lr=0.001,iternum=5000)
    #w[1:4] = 0
    print w.tolist()

    #w = None
    test_w(w,train_seq_list)
    test_w(w,valid_seq_list)
    
    
