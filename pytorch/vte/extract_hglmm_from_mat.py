import sys
from scipy.io import loadmat
import json
import glob
import numpy as np


annot_fn = '/home/datasets/coco/annotations/split/dataset_coco.json'
jobj = json.loads(file(annot_fn).read())

sent_id_lst = []
for image in jobj['images']:
    imgid = image['filename'].split('.')[0]
    sentnum = len(image['sentids'])
    sent_id_lst+=[imgid+'_'+str(i) for i in range(sentnum)]

mat_fn_lst = glob.glob('/home/datasets/coco/raw/annotation_text/hglmm/*.mat')
mat_fn_lst = sorted(mat_fn_lst, key=lambda fn:int(fn.split('/')[-1].split('.')[0].split('_')[-1]))

data = loadmat('pcamat.mat')
W = data['W']

outdir = '/home/datasets/coco/raw/annotation_text/hglmm_pca_npy/'
idx = 0
for fn in mat_fn_lst:
    data = loadmat(fn)
    feats = data['hglmm_30_ica_sent_vecs']
    featnum, samplenum = feats.shape
    feats = W.dot(feats)
    #print feats.shape
    for i in range(samplenum):
        feat = feats[:,i]
        sent_id = sent_id_lst[idx]
        print sent_id
        if 'train' in sent_id: sent_id = 'train/'+sent_id
        else: sent_id = 'val/'+sent_id
        outfn = outdir+sent_id+'.npy'
        print outfn
        np.save(outfn,feat)
        idx+=1
