import sys
import glob
import torch
import torch.utils.data as data
from skimage import io, transform
from PIL import Image
from torchvision import transforms, utils
import numpy as np

class COCOImageDataset(data.Dataset):
    
    def __init__(self,train_dir,val_dir,transform=None):
        self.train_fn_lst = glob.glob(train_dir+'/*.jpg')
        self.val_fn_lst = glob.glob(val_dir+'/*.jpg')
        self.transform = transform
        
    def __getitem__(self,index):
        if index<len(self.train_fn_lst):
            imgfn = self.train_fn_lst[index]
        else:
            imgfn = self.val_fn_lst[index-len(self.train_fn_lst)]
        #img = io.imread(imgfn)
        img = default_loader(imgfn)
        if self.transform:
            img = self.transform(img)
        return imgfn,img

    def __len__(self):
        return len(self.train_fn_lst)+len(self.val_fn_lst)

class COCOImageCropDataset(data.Dataset):

    def __init__(self,train_dir,val_dir,transform=None):
        self.train_fn_lst = glob.glob(train_dir+'/*.jpg')
        self.val_fn_lst = glob.glob(val_dir+'/*.jpg')
        self.transform = transform

    def __getitem__(self,index):
        if index<len(self.train_fn_lst):
            imgfn = self.train_fn_lst[index]
        else:
            imgfn = self.val_fn_lst[index-len(self.train_fn_lst)]
        img = default_loader(imgfn)
        imgs = crop10(img)
        imgs = [self.transform(img) for img in imgs]
        imgs = torch.stack(imgs,dim=0)
        return imgfn,imgs

    def __len__(self):
        return len(self.train_fn_lst)+len(self.val_fn_lst)

def crop10(img):
    img = img.resize((256,256))
    img_flip = img.transpose(Image.FLIP_LEFT_RIGHT)
    imgs = []
    imgs+=[img.crop((0, 0, 224, 224)), img.crop((0,32,224,256)), img.crop((32,0,256,224)), img.crop((32,32,256,256))]
    imgs+=[img_flip.crop((0, 0, 224, 224)), img_flip.crop((0,32,224,256)), img_flip.crop((32,0,256,224)), img_flip.crop((32,32,256,256))]
    imgs+=[img.crop((16,16,240,240)),img_flip.crop((16,16,240,240))]
    return imgs


###############
#  Transforms #
###############
def default_loader(path):
    return Image.open(path).convert('RGB')

def croped_transform():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
    ])
    return transform

def default_transform(size):
    transform = transforms.Compose([
    transforms.Scale(size),
    transforms.CenterCrop(size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], # resnet imagnet
        std=[0.229, 0.224, 0.225])
    ])
    return transform

class Rescale(object):
    def __init__(self, shape):
        self.shape = shape
        
    def __call__(self,img):
        img = transform.resize(img, self.shape)
        return img

class ToVGGTensor(object):
    def __init__(self):
        self.VGG_MEAN = np.array([0.485, 0.456, 0.406])
        self.VGG_STD = np.array([0.229, 0.224, 0.225])
    def __call__(self, img):
        img /=255.0
        img = (img - self.VGG_MEAN) / self.VGG_STD
        img = img.transpose((2,0,1)).astype('float32')
        return torch.from_numpy(img)

class COCOImgTextFeatDataset(data.Dataset):
    def __init__(self,idlst,img_feat_dir, text_feat_dir):
        self.img_feat_dir = img_feat_dir
        self.text_feat_dir = text_feat_dir
        #self.imgfeat_fnlst = glob.glob(img_feat_dir+'/*.npy')
        self.imgfeat_fnlst = [img_feat_dir+iid+'.npy' for iid in idlst]
    
    def __getitem__(self,idx):
        imgfn = self.imgfeat_fnlst[idx]
        imgfeat = np.load(imgfn)
        img_id = imgfn.split('/')[-1].split('.')[0]
        #text_fnlst = glob.glob(self.text_feat_dir+'/'+img_id+'*.npy')
        text_fnlst = [self.text_feat_dir+'/'+img_id+'_'+str(i)+'.npy' for i in range(5)]
        textfeat = np.array([np.load(text_fn) for text_fn in text_fnlst])
        return imgfn,imgfeat,textfeat

    def __len__(self):
        return len(self.imgfeat_fnlst)

class COCOImgFeatDataset(data.Dataset):
    def __init__(self,idlst,img_feat_dir):
        self.idlst = idlst
        self.img_feat_dir = img_feat_dir
    def __getitem__(self,idx):
        imgfeat = np.load(self.img_feat_dir+self.idlst[idx]+'.npy')
        return imgfeat
    def __len__(self):
        return len(self.idlst)

class COCOTextFeatDataset(data.Dataset):
    def __init__(self,idlst,text_feat_dir):
        self.idlst = idlst
        self.text_feat_dir = text_feat_dir
    def __getitem__(self,idx):
        text_fnlst = [self.text_feat_dir+'/'+self.idlst[idx]+'_'+str(i)+'.npy' for i in range(5)]
        textfeat = np.array([np.load(text_fn) for text_fn in text_fnlst])
        return textfeat
    def __len__(self):
        return len(self.idlst)

class COCOImgTextFeatPairDataset(data.Dataset):
    def __init__(self,idlst,img_feat_dir, text_feat_dir):
        self.img_feat_dir = img_feat_dir
        self.text_feat_dir = text_feat_dir
        self.idlst = idlst
        self.text_num = 5
    def __getitem__(self,idx):
        imgidx = idx/self.text_num
        imgfeat = np.load(self.img_feat_dir+self.idlst[imgidx]+'.npy')
        textidx = idx%self.text_num
        textfeat = np.load(self.text_feat_dir+self.idlst[imgidx]+'_'+str(textidx)+'.npy')
        return self.idlst[imgidx], textidx, imgfeat, textfeat
    def __len__(self):
        return self.text_num*len(self.idlst)
"""
Main Function
"""
def main():
    
    #mytrans = transforms.Compose([Rescale((224,224)),ToVGGTensor()])
    train_dir = '/home/datasets/coco/raw/train2014/'
    val_dir = '/home/datasets/coco/raw/val2014'
    dataset = COCOImageDataset(train_dir,val_dir,transform=default_transform(224))
    dataloader = data.DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)
    for i, batch in enumerate(dataloader):
        print len(batch[0])
    #dataset = COCOImageDataset(train_dir,val_dir,transform=None)

def main2():
    img_dir = '/home/datasets/coco/raw/vgg19_feat/train/'
    text_dir = '/home/datasets/coco/raw/annotation_text/hglmm_npy/train/'

    dataset = COCOImgTextFeatDataset(img_dir,text_dir)
    dataloader = data.DataLoader(dataset, batch_size=10, shuffle=True, num_workers=4)
    #imgfn, imgfeat, textfeat = dataset[0]
    #print type(imgfn)
    #print imgfeat.shape
    #print textfeat.shape
    
    for i, batch in enumerate(dataloader):
        imgfns, imgfeats, textfeats = batch
        idxs = [3,2,1,4]
        textfeats = textfeats.numpy()
        print textfeats.shape
        #textfeats = textfeats[0,idxs,:]
        #print textfeats.shape
        if i>3: break
    
if __name__=='__main__':
    main2()
