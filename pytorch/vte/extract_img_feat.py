import sys
from coco_dataset import *
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models

model = models.vgg19(pretrained=True)
new_classifier = nn.Sequential(*list(model.classifier.children())[:-1])
model.classifier = new_classifier

model.cuda()

mytrans = transforms.Compose([Rescale((224,224)),ToVGGTensor()])
train_dir = '/home/datasets/coco/raw/train2014/'
val_dir = '/home/datasets/coco/raw/val2014'
#dataset = COCOImageDataset(train_dir,val_dir,transform=default_transform(224))
dataset = COCOImageCropDataset(train_dir,val_dir,transform=croped_transform())
dataloader = data.DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)

for i, batch in enumerate(dataloader):
    
    fns,imgs = batch
    imgs = imgs.cuda()
    fns = list(fns)
    feats = []
    for j,fn in enumerate(fns):
        feat = model.forward(Variable(imgs[j,:]))
        feat = torch.mean(feat.data,dim=0)
        feats.append(feat)
    feats = torch.stack(feats,dim=0)
    feats = feats.cpu().numpy()
    """
    feats = model.forward(Variable(imgs))
    feats = feats.data.cpu().numpy()
    fns = list(fns)
    """

    for j,fn in enumerate(fns):
        feat = feats[j]
        outdir = '/home/datasets/coco/raw/vgg19_feat/'
        if 'train' in fn: outdir+='train/'
        else: outdir+='val/'
        outfn = outdir+fn.split('/')[-1].split('.')[0]+'.npy'
        print outfn
        np.save(outfn,feat)
