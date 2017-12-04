import sys
import numpy as np
import torch
from torch.autograd import Variable
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from coco_dataset import *
from model import *
from scipy.spatial.distance import cdist

def random_exclude(high,excludes=[]):
    out = np.random.randint(high)
    while out in excludes:
        out = np.random.randint(high)
    return out

def random_sample(batch):
    imgids, sentids, imgfeats, textfeats = batch

    ### image as anchor
    anchor_img, positive_text, negative_text = [],[],[]
    for i,iid in enumerate(imgids):
        for j,iid2 in enumerate(imgids):
            if iid!=iid2:
                anchor_img.append(imgfeats[i])
                positive_text.append(textfeats[i])
                negative_text.append(textfeats[j])
    anchor_img, positive_text, negative_text = torch.stack(anchor_img), torch.stack(positive_text), torch.stack(negative_text)

    ### text as anchof
    anchor_text, positive_img, negative_img = [],[],[]
    for i,iid in enumerate(imgids):
        for j,iid2 in enumerate(imgids):
            if iid!=iid2:
                anchor_text.append(textfeats[i])
                positive_img.append(imgfeats[i])
                negative_img.append(imgfeats[j])
    anchor_text, positive_img, negative_img = torch.stack(anchor_text), torch.stack(positive_img), torch.stack(negative_img)
    positive_text = positive_text.type(torch.FloatTensor)
    negative_text = negative_text.type(torch.FloatTensor)

    return anchor_img, positive_text, negative_text, anchor_text, positive_img, negative_img


"""
Input (batch) is 1500 (img, sent) pairs
"""
def hard_negative_sample(batch, img_net, text_net):
    K = 20
    imgids, sentids, imgfeats, textfeats = batch
    imgfeats, textfeats = imgfeats.numpy(), textfeats.numpy()
    img_emb = img_net(Variable(torch.Tensor(imgfeats).cuda())).data.cpu().numpy()
    text_emb = text_net(Variable(torch.Tensor(textfeats).cuda())).data.cpu().numpy()
    cos_sim_mat = img_emb.dot(text_emb.T)
    for i,iid in enumerate(imgids):
        for j,iid2 in enumerate(imgids):
            if iid==iid2: cos_sim_mat[i,j] = -1.1
    
    ### image as anchor
    anchor_img, positive_text, negative_text = [],[],[]
    for i, iid in enumerate(imgids):
        sorted_idxs = sorted(range(len(imgids)), key=lambda j:cos_sim_mat[i,j], reverse=True)
        anchor_img+=[imgfeats[i]]*K
        positive_text+=[textfeats[i]]*K
        for j in range(K): negative_text+=[textfeats[sorted_idxs[j]]]
    anchor_img, positive_text, negative_text = np.array(anchor_img), np.array(positive_text), np.array(negative_text)

    ### text as anchor
    anchor_text, positive_img, negative_img = [],[],[]
    for i,iid in enumerate(imgids):
        sorted_idxs = sorted(range(len(imgids)), key=lambda j:cos_sim_mat[j,i], reverse=True)
        for j in range(K): negative_img+=[imgfeats[sorted_idxs[j]]]
    negative_img = np.array(negative_img)
    anchor_text, positive_img = positive_text, anchor_img

    return anchor_img, positive_text, negative_text, anchor_text, positive_img, negative_img


#######################################
##           MAIN FUNCTION
#######################################

def main():
    #batch_size = 500
    batch_size = 128
    img_net = ImgBranch2()
    text_net = TextBranch2()
    #img_net, text_net = torch.load('img_net.pt'), torch.load('text_net.pt')
    tri_loss = TripletLoss(0.1)
    params = list(img_net.parameters())+list(text_net.parameters())
    opt = optim.SGD(params, lr=0.1, momentum=0.9, weight_decay=0.00005)
    scheduler = StepLR(opt, step_size=10, gamma=0.1)
    img_net.cuda()
    text_net.cuda()

    idlst = [l.strip() for l in file('train_ids.txt')]
    img_dir = '/home/datasets/coco/raw/vgg19_feat/train/'
    text_dir = '/home/datasets/coco/raw/annotation_text/hglmm_pca_npy/train/'

    dataset = COCOImgTextFeatPairDataset(idlst,img_dir,text_dir)
    dataloader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    ### Test set ###
    tiidlst = [l.strip() for l in file('test_ids.txt')]
    img_dir = '/home/datasets/coco/raw/vgg19_feat/val/'
    text_dir = '/home/datasets/coco/raw/annotation_text/hglmm_pca_npy/val/'
    img_feat_dataset = COCOImgFeatDataset(tiidlst, img_dir)
    text_feat_dataset = COCOTextFeatDataset(tiidlst,text_dir)

    ### train subset
    triidlst = [l.strip() for l in file('train_val_ids.txt')]
    img_dir = '/home/datasets/coco/raw/vgg19_feat/train/'
    text_dir = '/home/datasets/coco/raw/annotation_text/hglmm_pca_npy/train/'
    img_sub_dataset = COCOImgFeatDataset(triidlst, img_dir)
    text_sub_dataset = COCOTextFeatDataset(triidlst,text_dir)
    

    total_loss = 0
    for eidx in range(50):
        #total_loss = 0
        print 'epoch',eidx
        for i, batch in enumerate(dataloader):
            #anc_i, pos_t, neg_t, anc_t, pos_i, neg_i = hard_negative_sample(batch,img_net,text_net)
            anc_i, pos_t, neg_t, anc_t, pos_i, neg_i = random_sample(batch)
            sub_batch_num = 1
            sub_batch_size = anc_i.shape[0]/sub_batch_num
            for j in range(sub_batch_num):
                start, end = j*sub_batch_size, (j+1)*sub_batch_size
                anc_i_sub, pos_t_sub, neg_t_sub, neg_i_sub = anc_i[start:end], pos_t[start:end], neg_t[start:end], neg_i[start:end]

                #anc_i_sub = img_net(Variable(torch.Tensor(anc_i_sub).cuda()))
                #pos_t_sub = text_net(Variable(torch.Tensor(pos_t_sub).cuda()))
                #neg_t_sub = text_net(Variable(torch.Tensor(neg_t_sub).cuda()))
                #neg_i_sub = img_net(Variable(torch.Tensor(neg_i_sub).cuda()))
                anc_i_sub = img_net(Variable(anc_i_sub.cuda()))
                pos_t_sub = text_net(Variable(pos_t_sub.cuda()))
                neg_t_sub = text_net(Variable(neg_t_sub.cuda()))
                neg_i_sub = img_net(Variable(neg_i_sub.cuda()))
                anc_t_sub = pos_t_sub
                pos_i_sub = anc_i_sub
                loss1 = tri_loss(anc_i_sub, pos_t_sub, neg_t_sub)
                loss2 = tri_loss(anc_t_sub, pos_i_sub, neg_i_sub)
                loss = loss1+2*loss2

                opt.zero_grad()
                loss.backward()
                opt.step()
                total_loss+=loss.data[0]

            if i%200==0:
                print 'epoch',eidx,'batch', i
                test(tiidlst,img_feat_dataset, text_feat_dataset,img_net,text_net)
                print 'train sub'
                test(triidlst,img_sub_dataset, text_sub_dataset,img_net,text_net)
                print 'train loss:',total_loss
                total_loss = 0
            #break
        #break
        scheduler.step()

        ###### TEST ######
        # test(tiidlst,img_feat_dataset, text_feat_dataset,img_net,text_net)

        torch.save(img_net,'img_net.pt')
        torch.save(text_net,'text_net.pt')


###################################
# TEST FUNCTION
####################################
def test(tiidlst,img_feat_dataset,text_feat_dataset,img_net,text_net):
    img_net.eval()
    text_net.eval()
    img_emb, text_emb, text_ids = [],[],[]
    
    #print 'extract image embedding...'
    for i in range(len(img_feat_dataset)):
        feat = img_feat_dataset[i]
        feat = feat[np.newaxis,:]
        feat = img_net(Variable(torch.Tensor(feat).cuda()))
        feat = feat.data.cpu().numpy()
        img_emb.append(feat[0,:])
    img_emb = np.array(img_emb)
   

    #print 'extract text embedding...'
    for i in range(len(text_feat_dataset)):
        feat = text_feat_dataset[i]
        feat = text_net(Variable(torch.Tensor(feat).cuda()))
        feat = feat.data.cpu().numpy()
        text_emb+=feat.tolist()
        text_ids+=[tiidlst[i]]*5
    text_emb = np.array(text_emb)

    print 'calculating cosine similarity...'
    cos_sim_mat = img_emb.dot(text_emb.T)
    #cos_sim_mat = cdist(img_emb,text_emb,'cosine')
    print cos_sim_mat.shape

    imgnum, sentnum = cos_sim_mat.shape
    ### Text to Image
    r1,r5,r10,mrr = 0.0,0.0,0.0,0.0
    for i in range(sentnum):
        sorted_idxs = sorted(range(imgnum), key=lambda j:cos_sim_mat[j,i],reverse=True)
        sentid = text_ids[i]
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
    ### Image to Text
    img_net.train()
    text_net.train()

def main_test():
    img_net, text_net = torch.load('img_net.pt'), torch.load('text_net.pt')
    tiidlst = [l.strip() for l in file('test_ids.txt')]
    img_dir = '/home/datasets/coco/raw/vgg19_feat/val/'
    text_dir = '/home/datasets/coco/raw/annotation_text/hglmm_pca_npy/val/'
    img_feat_dataset = COCOImgFeatDataset(tiidlst, img_dir)
    text_feat_dataset = COCOTextFeatDataset(tiidlst,text_dir)
    test(tiidlst,img_feat_dataset,text_feat_dataset,img_net,text_net)

if __name__=='__main__':
    main()
    #main_test()
