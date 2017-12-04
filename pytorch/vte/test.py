import sys
import numpy as np
import json
from sklearn.cross_decomposition import CCA
import scipy.io as sio
import pickle


idlst = [l.strip() for l in file('train_ids.txt')]
img_dir = '/home/datasets/coco/raw/vgg19_feat/train/'
text_dir = '/home/datasets/coco/raw/annotation_text/hglmm_pca_npy/train/'
imgfeats, textfeats = [],[]
for iid in idlst:
    fn = img_dir+iid+'.npy'
    feat = np.load(fn)
    imgfeats.append(feat)
    
    text_fn = text_dir+iid+'_'+str(np.random.randint(5))+'.npy'
    feat = np.load(text_fn)
    textfeats.append(feat)

imgfeats = np.array(imgfeats)
textfeats = np.array(textfeats)
sio.savemat('cca_mat.mat',{'img':imgfeats,'text':textfeats})

imgmean = np.mean(imgfeats,axis=0)
textmean = np.mean(textfeats,axis=0)
sio.savemat('cca_mean.mat',{'img':imgmean,'text':textmean})

#data = sio.loadmat('cca_img.mat')
#print data['A'].shape

"""
fn = 'mRNN_split/ms_coco_test_list_mRNN.txt'
for l in file(fn):
    iid = l.split('/')[-1].split('.')[0]
    print iid
"""

