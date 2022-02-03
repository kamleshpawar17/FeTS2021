import torch
import glob
import os
import pickle
import random
import numpy as np
import nibabel as nb
import cv2
import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader
from patch_utils import find_bbox
from skimage.transform import resize
from albumentations import Compose, OneOf
from albumentations.augmentations.transforms import Blur, Flip, GridDistortion, MotionBlur, PadIfNeeded, RandomBrightness, GaussNoise, JpegCompression, RandomContrast, RandomGamma, RandomGridShuffle
from albumentations.augmentations.crops.transforms import CenterCrop, RandomResizedCrop 
from albumentations.augmentations.geometric.rotate import RandomRotate90, Rotate
from albumentations.augmentations.geometric.transforms import Affine, ElasticTransform, Perspective   
from albumentations.augmentations.geometric.resize import Resize


def dataAugumentation(H=224, W=224, p=0.5):
    '''
    Function to for data augmentation including geometric and intensity transforms
    '''
    return Compose([
        OneOf([Flip(p=1), RandomRotate90(p=1), Rotate(p=1), 
        RandomResizedCrop(p=1, height=H, width=W, scale=(0.5, 1.0), interpolation=cv2.INTER_NEAREST)], p=p),
        OneOf([Affine(scale=(0.75, 1.0), translate_percent=(0.0, 0.1), shear=(-20, 20), interpolation=cv2.INTER_LINEAR, p=1), Perspective(scale=(0.05, 0.1), interpolation=cv2.INTER_LINEAR, p=1), GridDistortion(p=1)], p=p),
        OneOf([Blur(p=1), MotionBlur(p=1)], p=p),
        RandomContrast(limit=0.2, always_apply=False, p=p),
        ], additional_targets={'x_t2':'image', 'x_t1ce':'image', 'x_flair':'image'}, p=p)
        
        
class SliceDataModule(pl.LightningDataModule):
    '''
    SliceDataModule: PyTorch lightning data module for first stage network 
    '''
    def __init__(self, train_path, val_path, batch_size=4, num_workers=8, img_sz=224, data_aug=False, p=0.8):
        '''
        train_path: path to training data (numpy files) 
        val_path: path to validation data (numpy files) 
        batch_size: batch size for training
        num_workers: (int) number of workers for data generator for multiprocessing
        img_sz: (int) image size, a 2D image will be either cropped of zeros padded to make (img_sz x img_sz) image.
        data_aug: bool, True: apply data augmentation, False: do not apply data augmentation
        p: probablity of applying data augmentation
        '''
        super().__init__()
        self.train_path = train_path
        self.val_path = val_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_sz = img_sz
        self.data_aug = data_aug
        self.p = p
        self.list_train = glob.glob(self.train_path)
        self.list_val = glob.glob(self.val_path)
        
    def prepare_data(self):
        return 0

    def setup(self, stage: str = None):
        return 0        

    def train_dataloader(self):
        self.data_train = SliceDataset(list_data=self.list_train, img_sz=self.img_sz, data_aug=self.data_aug, p=self.p)
        return DataLoader(self.data_train, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        self.data_val = SliceDataset(list_data=self.list_val, img_sz=self.img_sz, data_aug=False, p=self.p)
        return DataLoader(self.data_val, batch_size=self.batch_size, num_workers=self.num_workers)

class SliceDataset(torch.utils.data.Dataset):
    def __init__(self, list_data, img_sz=224, data_aug=False, p=0.8):
        self.img_sz = img_sz
        self.data_aug = data_aug
        self.data_instance = list_data
        self.aug = dataAugumentation(H=img_sz, W=img_sz, p=p)
        print('Number of images: ', len(self.data_instance))
            
    def __len__(self):
        return len(self.data_instance)
    
    def zpad_crop_to_n(self, x, n=196):
        '''
        takes 3D image and decides to either crop or pad first two dimension to make the first two dimension nxn
        '''
        d, r, c = x.shape
        if r > n:
            start = int((r-n)/2.)
            x = x[:, start:start+n, :]
        else:
            zpad_r0 = int(np.ceil((n-r) / 2.))
            zpad_r1 = int(np.floor((n-r) / 2.)) 
            x = np.pad(x, ((0, 0), (zpad_r0, zpad_r1), (0, 0)), 'constant')
        if c > n:
            start = int((c-n)/2.)
            x = x[:, :, start:start+n]
        else:
            zpad_r0 = int(np.ceil((n-c) / 2.))
            zpad_r1 = int(np.floor((n-c) / 2.)) 
            x = np.pad(x, ((0, 0), (0, 0), (zpad_r0, zpad_r1)), 'constant')
        return x
    
    def normalize_mean_std(self, x):
        xmean = np.mean(x, axis=(-1, -2), keepdims=True)
        xstd = np.std(x, axis=(-1, -2), keepdims=True)
        y = (x - xmean)/(1e-7+xstd)
        y[y>6.0] = 6.0
        y[y<-6.0] = -6.0
        return y
    
    def augment(self,x_t1, x_t2, x_t1ce, x_flair, x_seg):
        augmented1 = self.aug(image=x_t1, x_t1=x_t1, x_t2=x_t2, x_t1ce=x_t1ce, x_flair=x_flair, mask=x_seg)
        x_t1_new, x_t2_new, x_t1ce_new, x_flair_new, x_seg_new = augmented1['image'], augmented1['x_t2'], augmented1['x_t1ce'], augmented1['x_flair'], augmented1['mask']
        return x_t1_new, x_t2_new, x_t1ce_new, x_flair_new, x_seg_new
        
    def get_batchsize_one(self, i):
        '''
        takes index and generates one sample of input data 
        '''
        fname = self.data_instance[i]
        inp_img = np.load(fname)
        x_inp = self.zpad_crop_to_n(inp_img[0:4, :, :], n=self.img_sz)
        x_seg = self.zpad_crop_to_n(inp_img[4:5, :, :], n=self.img_sz)
        x_seg[x_seg==4] = 3
        return x_inp.astype('float32'), x_seg.astype('int')
         
    def __getitem__(self, i: int):
        x, y = self.get_batchsize_one(i)
        if self.data_aug:
            x_flair, x_t1, x_t2, x_t1ce, x_seg = x[0], x[1], x[2], x[3], y[0].astype('int')
            x_t1, x_t2, x_t1ce, x_flair, x_seg = self.augment(x_t1, x_t2, x_t1ce, x_flair, x_seg)
            x_t1, x_t2, x_t1ce, x_flair, y  = np.expand_dims(x_t1, 0),  np.expand_dims(x_t2, 0),  np.expand_dims(x_t1ce, 0),  np.expand_dims(x_flair, 0),  np.expand_dims(x_seg, 0).astype('int')
            x = np.concatenate((x_flair, x_t1, x_t2, x_t1ce), 0)
        x = self.normalize_mean_std(x)
        return (x, y)

class PatchDataModule(pl.LightningDataModule):
    def __init__(self, train_path, val_path, batch_size=4, num_workers=8, img_sz=224, data_aug=False, p=0.8, minbbox=56, bboxper=1.25, min_num_pixel=64, use_cache=False):
        '''
        train_path: path to training data (numpy files) 
        val_path: path to validation data (numpy files) 
        batch_size: batch size for training
        num_workers: (int) number of workers for data generator for multiprocessing
        img_sz: (int) image size, a 2D image will be either cropped of zeros padded to make (img_sz x img_sz) image.
        data_aug: bool, True: apply data augmentation, False: do not apply data augmentation
        p: probablity of applying data augmentation
        minbbox: size of minimum bounding box allowed for cropping tumor region
        bboxper: size of cropping compared to bounding box 1 means same size, 1.25 means crop 25% more than the actual bounding box
        min_num_pixel: number of positive pixels for a region to be consireded as bounding box
        use_cache: use the cached file for faster processing, set it to True if using the same dataset else set it to False
        '''
        super().__init__()
        self.train_path = train_path
        self.val_path = val_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_sz = img_sz
        self.data_aug = data_aug
        self.p = p
        self.minbbox = minbbox
        self.bboxper = bboxper
        self.min_num_pixel = min_num_pixel
        self.list_train = glob.glob(self.train_path)
        self.list_val = glob.glob(self.val_path)
        self.use_cache = use_cache
        
    def prepare_data(self):
        return 0

    def setup(self, stage: str = None):
        return 0        

    def train_dataloader(self):
        self.data_train = PatchDataset(list_data=self.list_train, img_sz=self.img_sz, data_aug=self.data_aug, p=self.p, minbbox=self.minbbox, bboxper=self.bboxper, min_num_pixel=self.min_num_pixel, split='train_', use_cache=self.use_cache)
        return DataLoader(self.data_train, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        self.data_val = PatchDataset(list_data=self.list_val, img_sz=self.img_sz, data_aug=False, p=self.p, minbbox=self.minbbox, bboxper=self.bboxper, min_num_pixel=self.min_num_pixel, split='val_', use_cache=self.use_cache)
        return DataLoader(self.data_val, batch_size=self.batch_size, num_workers=self.num_workers)
        
class PatchDataset(torch.utils.data.Dataset):
    def __init__(self, list_data, img_sz=224, data_aug=False, p=0.8, minbbox=56, bboxper=1.25, min_num_pixel=64, split='train_', use_cache=False):
        self.img_sz = img_sz
        self.data_aug = data_aug
        self.resize = Resize(img_sz, img_sz, cv2.INTER_NEAREST) 
        self.aug = dataAugumentation(H=img_sz, W=img_sz, p=p)
        self.minbbox = minbbox
        self.bboxper = bboxper
        self.min_num_pixel = min_num_pixel
        self.chche_name = split+'cache.pkl'
        self.use_cache = use_cache
        self.data_instance = self.find_patch_list(list_data)
        print('Number of images: ', len(self.data_instance))
            
    def __len__(self):
        return len(self.data_instance)
        
    def find_patch_list(self, list_data):
        if os.path.exists(self.chche_name) and self.use_cache:
            with open(self.chche_name, "rb") as output_file:
                data_instance = pickle.load(output_file)
        else:
            data_instance = []
            for fname in list_data:
                img = np.load(fname)
                if np.sum(img[4, :, :]>0) > self.min_num_pixel:
                    bbox_list = find_bbox(img[4, :, :], minbbox=self.minbbox, bboxper=self.bboxper)
                    for bbox in bbox_list:
                        data_instance.append([fname, bbox])
            with open(self.chche_name, "wb") as output_file:
                pickle.dump(data_instance, output_file)
        return data_instance
        
    def normalize_mean_std(self, x):
        xmean = np.mean(x, axis=(-1, -2), keepdims=True)
        xstd = np.std(x, axis=(-1, -2), keepdims=True)
        y = (x - xmean)/(1e-7+xstd)
        y[y>6.0] = 6.0
        y[y<-6.0] = -6.0
        return y
    
    def augment(self,x_t1, x_t2, x_t1ce, x_flair, x_seg):
        augmented1 = self.aug(image=x_t1, x_t1=x_t1, x_t2=x_t2, x_t1ce=x_t1ce, x_flair=x_flair, mask=x_seg)
        x_t1_new, x_t2_new, x_t1ce_new, x_flair_new, x_seg_new = augmented1['image'], augmented1['x_t2'], augmented1['x_t1ce'], augmented1['x_flair'], augmented1['mask']
        return x_t1_new, x_t2_new, x_t1ce_new, x_flair_new, x_seg_new
        
    def get_batchsize_one(self, i):
        '''
        takes index and generates one sample of input data 
        '''
        fname, bbox = self.data_instance[i]
        minr, minc, maxr, maxc = bbox
        inp_temp = np.load(fname)
        inp_img = np.zeros((5, self.img_sz, self.img_sz))
        for k in range(5):
            inp_img[k, :, :] = self.resize(image=inp_temp[k, minr:maxr, minc:maxc])['image']
        x_inp = inp_img[0:4, :, :]
        x_seg = inp_img[4:5, :, :]
        x_seg[x_seg==4] = 3
        return x_inp.astype('float32'), x_seg.astype('int')
         
    def __getitem__(self, i: int):
        x, y = self.get_batchsize_one(i)
        if self.data_aug:
            x_flair, x_t1, x_t2, x_t1ce, x_seg = x[0], x[1], x[2], x[3], y[0].astype('int')
            x_t1, x_t2, x_t1ce, x_flair, x_seg = self.augment(x_t1, x_t2, x_t1ce, x_flair, x_seg)
            x_t1, x_t2, x_t1ce, x_flair, y  = np.expand_dims(x_t1, 0),  np.expand_dims(x_t2, 0),  np.expand_dims(x_t1ce, 0),  np.expand_dims(x_flair, 0),  np.expand_dims(x_seg, 0).astype('int')
            x = np.concatenate((x_flair, x_t1, x_t2, x_t1ce), 0)
        x = self.normalize_mean_std(x)
        return (x, y)
