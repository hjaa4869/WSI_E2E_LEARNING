from __future__ import print_function, division
import os
import torch
import random
import numpy as np
import pandas as pd
import math
import re
import pdb
import pickle
import openslide
from scipy import stats
#import cvfunctional as F1
#import function_code.many_channels
from torch.utils.data import Dataset
import h5py
import lmdb
from datasets.data_utils import imfrombytes
from utils.utils import generate_split, nth
from torch.utils.data import get_worker_info
import time

def eval_transforms(pretrained=False):
    if pretrained:
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)

    else:
        mean = (0.5,0.5,0.5)
        std = (0.5,0.5,0.5)

    trnsfrms_val = transforms.Compose(
                    [
                     transforms.ToTensor(),
                     transforms.Normalize(mean = mean, std = std)
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

class Generic_WSI_Classification_Dataset(Dataset):
    def __init__(self,
        csv_path = './dataset_csv/camelyon16.csv',
        shuffle = False, #控制打乱数据
        label_col = 'label',
        print_info = True,#控制打印信息                 
        seed = 7,         
        num_classes = 2
        ):
        self.num_classes = num_classes
        self.seed = seed
        self.print_info = print_info
        self.train_ids, self.val_ids, self.test_ids  = (None, None, None)
        self.data_dir = None
        self.label_col = label_col
        slide_data = pd.read_csv(csv_path)

        #打乱数据
        if shuffle:
            # np.random.seed(seed)
            # np.random.shuffle(slide_data)
            slide_data = slide_data.sample(
                    frac=1,
                    random_state=seed
                ).reset_index(drop=True)
        self.slide_data = slide_data

        self.cls_ids_prep()

        if print_info:
            self.summarize()

    def cls_ids_prep(self):#各类别对应哪些slide       
        self.slide_cls_ids = [[] for i in range(self.num_classes)]
        for i in range(self.num_classes):
            self.slide_cls_ids[i] = np.where(self.slide_data['label'] == i)[0]

    def __len__(self):
        return len(self.slide_data)

    def summarize(self):
        print("label column: {}".format(self.label_col))
        print("number of classes: {}".format(self.num_classes))
        print("slide-level counts: ", '\n', self.slide_data['label'].value_counts(sort = False))
        for i in range(self.num_classes):
            print('Slide-LVL; %d: %d' % (i, self.slide_cls_ids[i].shape[0]))

#下面两个函数是最核心的划分和返还不同数据集过程
            
    def get_split_from_df(self, all_splits, split_key='train'):
        split = all_splits[split_key]
        split = split.dropna().reset_index(drop=True)

        if len(split) > 0:
            mask = self.slide_data['slide_id'].isin(split.tolist())
            df_slice = self.slide_data[mask].reset_index(drop=True)
            split = Generic_Split(df_slice)
        else:
            split = None
        
        return split

    def return_splits(self, csv_path=None):
        
        assert csv_path 
        all_splits = pd.read_csv(csv_path, dtype=self.slide_data['slide_id'].dtype)  
        train_split = self.get_split_from_df(all_splits, 'train')
        val_split = self.get_split_from_df(all_splits, 'val')
        test_split = self.get_split_from_df(all_splits, 'test')
            
        return train_split, val_split, test_split

    def get_list(self, ids):
        return self.slide_data['slide_id'][ids]

    def getlabel(self, ids):
        return self.slide_data['label'][ids]

    def __getitem__(self, idx):
        return None

    # def save_split(self, filename):
    #     train_split = self.get_list(self.train_ids)
    #     val_split = self.get_list(self.val_ids)
    #     test_split = self.get_list(self.test_ids)
    #     df_tr = pd.DataFrame({'train': train_split})
    #     df_v = pd.DataFrame({'val': val_split})
    #     df_t = pd.DataFrame({'test': test_split})
    #     df = pd.concat([df_tr, df_v, df_t], axis=1) 
    #     df.to_csv(filename, index = False)
        
#检查是否数据泄露等情况发生
#     def test_split_gen(self, return_descriptor=False):

#         if return_descriptor:
#             index = [list(self.label_dict.keys())[list(self.label_dict.values()).index(i)] for i in range(self.num_classes)]
#             columns = ['train', 'val', 'test']
#             df = pd.DataFrame(np.full((len(index), len(columns)), 0, dtype=np.int32), index= index,
#                             columns= columns)

#         count = len(self.train_ids)
#         print('\nnumber of training samples: {}'.format(count))
#         labels = self.getlabel(self.train_ids)
#         unique, counts = np.unique(labels, return_counts=True)
#         for u in range(len(unique)):
#             print('number of samples in cls {}: {}'.format(unique[u], counts[u]))
#             if return_descriptor:
#                 df.loc[index[u], 'train'] = counts[u]
        
#         count = len(self.val_ids)
#         print('\nnumber of val samples: {}'.format(count))
#         labels = self.getlabel(self.val_ids)
#         unique, counts = np.unique(labels, return_counts=True)
#         for u in range(len(unique)):
#             print('number of samples in cls {}: {}'.format(unique[u], counts[u]))
#             if return_descriptor:
#                 df.loc[index[u], 'val'] = counts[u]

#         count = len(self.test_ids)
#         print('\nnumber of test samples: {}'.format(count))
#         labels = self.getlabel(self.test_ids)
#         unique, counts = np.unique(labels, return_counts=True)
#         for u in range(len(unique)):
#             print('number of samples in cls {}: {}'.format(unique[u], counts[u]))
#             if return_descriptor:
#                 df.loc[index[u], 'test'] = counts[u]

#         assert len(np.intersect1d(self.train_ids, self.test_ids)) == 0
#         assert len(np.intersect1d(self.train_ids, self.val_ids)) == 0
#         assert len(np.intersect1d(self.val_ids, self.test_ids)) == 0

#         if return_descriptor:
#             return df


class Generic_MIL_Dataset(Generic_WSI_Classification_Dataset):
    def __init__(self,
        data_dir, 
        use_lmdb,
        lmdb_path,
        sampled_number,
        **kwargs):
        self.current_epoch = 0
        super(Generic_MIL_Dataset, self).__init__(**kwargs)
        self.data_dir = data_dir
        self.use_lmdb = use_lmdb
        self.lmdb_path = lmdb_path
        self.sampled_number = sampled_number
            
    def _init_lmdb(self):
        self.env = lmdb.open(
            self.lmdb_path,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            map_size=100 * (1024**3)
        )
        with self.env.begin(write=False) as txn:
            self.pn_dict = pickle.loads(txn.get(b'__pn__'))

    def set_epoch(self, epoch):
        self.current_epoch = epoch
            
    def __getitem__(self, idx):
        if not hasattr(self, "env"):
            self._init_lmdb()
        slide_id = self.slide_data['slide_id'][idx]
        label = self.slide_data['label'][idx]
        if type(self.data_dir) == dict:
            source = self.slide_data['source'][idx]
            data_dir = self.data_dir[source]
        else:
            data_dir = self.data_dir

        if not self.use_lmdb:
            full_path = os.path.join(data_dir,'h5_files','{}.h5'.format(slide_id))
            with h5py.File(full_path,'r') as hdf5_file:
                features = hdf5_file['features'][:]
                coords = hdf5_file['coords'][:]
            features = torch.from_numpy(features)
            return features, label#, coords
                
                
        else:
            slide_name = slide_id
            with self.env.begin(write=False) as txn:
                if slide_name not in self.pn_dict:
                    raise ValueError(f"Slide ID {slide_name} not found in LMDB metadata.")
                num_patches = self.pn_dict[slide_name]
                
                
                worker_info = get_worker_info()
                worker_id = worker_info.id if worker_info else 0
                
                
                patches_tensor = torch.empty((self.sampled_number, 3, 256, 256), dtype=torch.float32)
                features_tensor = torch.empty((self.sampled_number, 1024), dtype=torch.float32)
                
                #hash每次好像是会随机生成的，想seed稳定复现还是得再看一下
                seed = hash((self.current_epoch, worker_id, slide_name)) % (2**32)
                rng = random.Random(seed)

                patch_ids = [f"{slide_name}-{i}" for i in range(num_patches)]
                rng.shuffle(patch_ids)

                if num_patches >= self.sampled_number:
                    patch_ids = patch_ids[:self.sampled_number]
                else:
                    patch_ids = patch_ids + rng.choices(patch_ids, k=self.sampled_number - num_patches)
                
                for i, key_str in enumerate(patch_ids):
                    bytes = txn.get(key_str.encode('ascii'))
                    if bytes is None:
                        raise KeyError(f"LMDB missing key: {key_str}")
                    value = pickle.loads(bytes)
                    img_array = imfrombytes(value['img'])
                    feature = value['feature']
                    
                    patches_tensor[i] = torch.from_numpy(
                        img_array.transpose(2,0,1)
                    ).float()

                    features_tensor[i] = torch.from_numpy(
                        feature
                    ).float()
            
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1) * 255
            std = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1) * 255
            patches_tensor = (patches_tensor - mean) / std
            
            return patches_tensor, features_tensor, label
                

class Generic_Split(Generic_MIL_Dataset):
    def __init__(self, slide_data, 
        data_dir="./data_feat/Camelyon16_level2", 
        num_classes=2,  
        use_lmdb=True,
        lmdb_path='./lmdb_feat/camelyon16_256_level2.lmdb',
        sampled_number=128):
        self.current_epoch = 0
        self.use_h5 = True
        self.slide_data = slide_data
        self.data_dir = data_dir
        self.num_classes = num_classes
        self.slide_cls_ids = [[] for i in range(self.num_classes)]
        self.use_lmdb = use_lmdb
        self.lmdb_path = lmdb_path
        self.sampled_number = sampled_number
        for i in range(self.num_classes):
            self.slide_cls_ids[i] = np.where(self.slide_data['label'] == i)[0]

    def __len__(self):
        return len(self.slide_data)

#在第二级上把参数定义好，然后传入第一级，第一级再传入第三级应该就好了/或者试试用super父类，看能不能继承参数
#还是算了，不要super了，怕参数之间互相覆盖；现在的逻辑是，dataset首先创建完整的数据集，此时的slide_data就是完整的slide序列，而后在调用函数划分数据集的时候，每个子集（训练、测试、验证）就会重新返还一个类，这个类只传入了slide_data的信息，不过这个slidedata和前一个完整版的不一样，这个只包含其各自的序列（训练、测试或是验证），最终得到三个dataset，每个都是相当于第二个类的效果，不过参数传入是需要在这里手动定义的，建议之后都train的py文件和这里进行一下手动定义参数



#这里注释掉的是lmdb文件中只有img的情景，正在使用的是lmdb有img以及特征的情景
#             slide_name = slide_id
#             with self.env.begin(write=False) as txn:
#                 if slide_name not in self.pn_dict:
#                     raise ValueError(f"Slide ID {slide_name} not found in LMDB metadata.")
#                 num_patches = self.pn_dict[slide_name]
#                 worker_info = get_worker_info()
#                 worker_id = worker_info.id if worker_info else 0
#                 seed = self.current_epoch * 10 + worker_id
#                 rng = random.Random(seed)
#                 sampled_indices = rng.choices(
#                     range(num_patches),
#                     k=self.sampled_number
#                 )
#                 patch_ids = [f"{slide_name}-{idx}" for idx in sampled_indices]
#                 patches_tensor = torch.empty((self.sampled_number, 3, 256, 256), dtype=torch.float32)
#                 for i, key_str in enumerate(patch_ids):
#                     patch_bytes = txn.get(key_str.encode('ascii'))
#                     if patch_bytes is None:
#                         raise KeyError(f"LMDB missing key: {key_str}")
#                     img_array = imfrombytes(pickle.loads(patch_bytes))
#                     patches_tensor[i] = torch.from_numpy(img_array.transpose(2, 0, 1))
                    
#             patches_tensor = patches_tensor.float()
#             if patches_tensor.max() <= 1.5:
#                 mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1)
#                 std = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1)
#             else:
#                 mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1) * 255
#                 std = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1) * 255
#             patches_tensor = (patches_tensor - mean) / std
#             return patches_tensor, label