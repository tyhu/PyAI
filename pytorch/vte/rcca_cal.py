import sys
import os
import numpy as np
sys.path.append('/home/harry/github/pyrcca/')
import rcca

if os.path.isfile('img_train.npy'):
    imgfeats = np.load('img_train.npy')
    textfeats = np.load('text_train.npy')
else:

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

    np.save('img_train.npy',imgfeats)
    np.save('text_train.npy',textfeats)

print 'CCA training...'
cca = rcca.CCA(kernelcca = False, reg = 0.1, numCC = 32)
cca.train([imgfeats,textfeats])
print type(cca.comps)
"""
data = sio.loadmat('cca_mean.mat')
imgmean = data['img']
textmean = data['text']

tiidlst = [l.strip() for l in file('test_ids.txt')]
img_dir = '/home/datasets/coco/raw/vgg19_feat/val/'
text_dir = '/home/datasets/coco/raw/annotation_text/hglmm_pca_npy/val/'

ttiidlst = []
imgfeats,textfeats = [],[]

for iid in tiidlst:
    ttiidlst+=[iid]*5
    img_feat_fn = img_dir+iid+'.npy'
    text_feat_fnlst = [text_dir+iid+'_'+str(i)+'.npy' for i in range(5)]
    imgfeats.append(np.load(img_feat_fn))
    textfeats+=[np.load(fn) for fn in text_feat_fnlst]

imgfeats,textfeats = np.array(imgfeats),np.array(textfeats)

imgnum, textnum = len(tiidlst),len(ttiidlst)
imgfeats = imgfeats - np.tile(imgmean,(imgnum,1))
textfeats = textfeats - np.tile(textmean,(textnum,1))
imgfeats, textfeats = imgfeats.dot(img_transform), textfeats.dot(text_transform)
"""
