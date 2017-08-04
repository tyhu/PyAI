import sys
import numpy as np
from myconfig import *
from util_track import *
from mot_tracker import *

codelst = colorcodes()
display = False
for seq in train_seq_list:
    print 'Tracking',seq
    tracker = Tracker()
    #detfn = datadir+'/'+seq+'/det/det.txt'
    detfn = '/home/tingyaoh/github/sort/data/'+seq+'/det.txt'
    imdir = datadir+'/'+seq+'/img1/'
    gtfn = datadir+'/'+seq+'/gt/gt.txt'
    #plot_detection(gtfn,imdir)
    
    if not display: outfile = file('output/'+seq+'.txt','w')
    for frid,img,patchlst,dets in yieldMOTImgDet(detfn,imdir):
        if frid==1:
            tracker.processFirstImgNDets(img,dets)
        else:
            tracker.processNextImgNDets(img,dets)
        if not display: tracker.output(outfile)
        if display:
            tracker.plot(img,codelst)
            cv2.waitKey(-1)
    
