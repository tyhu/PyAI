import sys
from coco_dataset import *
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models

model = models.resnet152(pretrained=True)
class ResNetMaxPool(nn.Module):
    def __init__(self):
        super(ResNetMaxPool, self).__init__()
        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4
        self.avgpool = model.avgpool

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return x

extractor = ResNetMaxPool()
extractor.cuda()

train_dir = '/home/datasets/coco/raw/train2014/'
val_dir = '/home/datasets/coco/raw/val2014'
dataset = COCOImageDataset(train_dir,val_dir,transform=default_transform(224))
dataloader = data.DataLoader(dataset, batch_size=12, shuffle=False, num_workers=4)

for i, batch in enumerate(dataloader):

    fns,imgs = batch
    imgs = imgs.cuda()
    feats = extractor.forward(Variable(imgs))
    feats = feats.data.cpu().numpy()
    fns = list(fns)
    print feats.shape
    break
