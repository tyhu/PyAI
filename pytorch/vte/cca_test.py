import numpy as np
import scipy.io as sio

tiidlst = [l.strip() for l in file('test_ids.txt')]
img_dir = '/home/datasets/coco/raw/vgg19_feat/val/'
text_dir = '/home/datasets/coco/raw/annotation_text/hglmm_pca_npy/val/'

data = sio.loadmat('cca_mean.mat')
imgmean = data['img']
textmean = data['text']

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
img_transform = sio.loadmat('cca_img.mat')['A']
text_transform = sio.loadmat('cca_text.mat')['B']
imgfeats, textfeats = imgfeats.dot(img_transform), textfeats.dot(text_transform)

sim_mat = imgfeats.dot(textfeats.T)
    
imgnum, sentnum = sim_mat.shape
### Text to Image
r1,r5,r10,mrr = 0.0,0.0,0.0,0.0
for i in range(sentnum):
    sorted_idxs = sorted(range(imgnum), key=lambda j:sim_mat[j,i],reverse=True)
    sentid = ttiidlst[i]
    r1_list = [tiidlst[j] for j in sorted_idxs[:1]]
    r5_list = [tiidlst[j] for j in sorted_idxs[:5]]
    r10_list = [tiidlst[j] for j in sorted_idxs[:10]]
    all_list = [tiidlst[j] for j in sorted_idxs]
    if sentid in r1_list: r1+=1
    if sentid in r5_list: r5+=1
    if sentid in r10_list: r10+=1
    first_idx = all_list.index(sentid)+1
    mrr+=1.0/first_idx
    #print sentid, r10_list
r1/=sentnum
r5/=sentnum
r10/=sentnum
mrr/=sentnum
print 'r1:',r1,'r5:',r5,'r10:',r10, 'mrr:',mrr
