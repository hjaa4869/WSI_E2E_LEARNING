from __future__ import print_function, division
import os
import torch
import numpy as np
import pandas as pd
import math
import re
import cv2
import pdb
import pickle
import cvfunctional as F1
from torch.utils.data import Dataset, DataLoader, sampler
from torchvision import transforms, utils, models
import torch.nn.functional as F
#import function_code.many_channels
from PIL import Image
import h5py
from random import randrange

def eval_transforms(pretrained=False):
    if pretrained:
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)

    else:
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)

    trnsfrms_val = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ]
    )

    return trnsfrms_val

def simple_transforms(pretrained=False):
    trnsfrms_val = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    return trnsfrms_val

class Dataset_All_Bags(Dataset):

    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.df['slide_id'][idx]    
    
class Dataset_All_Bags_real(Dataset):#该处理可以跳过h5目录中不存在的文件数据
    def __init__(self, csv_path, h5_path,args):
        self.df = pd.read_csv(csv_path)
        files = os.listdir(h5_path)
        slide_ids = [os.path.splitext(file)[0] for file in files]
        self.df = self.df[self.df['slide_id'].isin(slide_ids)].reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.df['slide_id'][idx]
    
class Whole_Slide_Bag_FP(Dataset):
    def __init__(self,
                 file_path,
                 wsi,
                 pretrained=False,
                 custom_transforms=None
                 ):
        self.pretrained = pretrained
        self.wsi = wsi
        if not custom_transforms:
            self.roi_transforms = eval_transforms(pretrained=pretrained)
        else:
            self.roi_transforms = custom_transforms

        self.file_path = file_path

        with h5py.File(self.file_path, "r") as f:
            dset = f['coords']
            self.patch_level = f['coords'].attrs['patch_level']
            self.patch_size = f['coords'].attrs['patch_size']
            self.length = len(dset)
        self.summary()

    def __len__(self):
        return self.length

    def summary(self):
        
        hdf5_file = h5py.File(self.file_path, "r")
        dset = hdf5_file['coords']
        for name, value in dset.attrs.items():
            print(name, value)

        print('\nfeature extraction settings')
        print('pretrained: ', self.pretrained)
        print('transformations: ', self.roi_transforms)

    def __getitem__(self, idx):
        with h5py.File(self.file_path, 'r') as hdf5_file:
            coord = np.asarray(hdf5_file['coords'][idx])

        temp_patch_level, temp_patch_size = self.patch_level, self.patch_size

        while True:
            try:
                img = self.wsi.read_region(coord, temp_patch_level, (temp_patch_size, temp_patch_size)).convert('RGB')
                break
            except:
                temp_patch_level, temp_patch_size = temp_patch_level + 1, temp_patch_size // 2
                print('index error:', idx)

        img = self.roi_transforms(img).unsqueeze(0)

        return img, coord


