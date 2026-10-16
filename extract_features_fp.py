import torch
import torch.nn as nn
from math import floor
import os
import random
import numpy as np
import pdb
import time
import timm
from torchvision import transforms
from datasets.dataset_h5 import Dataset_All_Bags, Whole_Slide_Bag_FP#, Whole_Slide_Bag_FP_DCT#, Whole_Slide_Bag_FP_Distort
from torch.utils.data import DataLoader
from models.resnet_custom import resnet50_baseline
# from models.model_clam import RES_50
import argparse
from utils.utils import print_network, collate_features, collate_features_dct
from utils.file_utils import save_hdf5
from PIL import Image
import h5py
import openslide
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def compute_w_loader(file_path, output_path, wsi, model,
                     batch_size=8, verbose=0, print_every=20, pretrained=False,
                     custom_transforms=None, just_patches=False):
    # dataset = Whole_Slide_Bag_FP_Distort(file_path=file_path, wsi=wsi, pretrained=pretrained)
    #注释部分用于增加图像上的扰动,其中部分参数被优化
    dataset = Whole_Slide_Bag_FP(file_path=file_path, wsi=wsi,
                                 pretrained=pretrained,custom_transforms=custom_transforms)
    kwargs = {'num_workers': 4, 'pin_memory': True} if device.type == "cuda" else {}
    loader = DataLoader(dataset=dataset, batch_size=batch_size, **kwargs, collate_fn=collate_features)

    if verbose > 0:
        print('processing {}: total of {} batches'.format(file_path, len(loader)))

    mode = 'w'
    for count, (batch, coords) in enumerate(loader):
        with torch.no_grad():
            if count % print_every == 0:
                print('batch {}/{}, {} files processed'.format(count, len(loader), count * batch_size))
        batch = batch.to(device, non_blocking=True)

        features = model(batch).detach().cpu().numpy()
        
        asset_dict = {'features': features, 'coords': coords}
        save_hdf5(output_path, asset_dict, attr_dict=None, mode=mode)
        mode = 'a'

    return output_path

def compute_w_loader_dct(file_path, output_path, wsi,
                     batch_size=8, verbose=0, print_every=20, pretrained=False,
                         custom_transforms=None, label_file_path = None, only_y = False):
    dataset = Whole_Slide_Bag_FP_DCT(file_path=file_path, wsi=wsi, pretrained=pretrained,
                                     custom_transforms=custom_transforms, label_file_path = label_file_path, only_y = only_y)
    kwargs = {'num_workers': 4, 'pin_memory': True} if device.type == "cuda" else {}
    loader = DataLoader(dataset=dataset, batch_size=batch_size, **kwargs, collate_fn=collate_features_dct)

    if verbose > 0:
        print('processing {}: total of {} batches'.format(file_path, len(loader)))

    mode = 'w'
    for count, (dct_y, coord, label) in enumerate(loader):
        with torch.no_grad():
            if count % print_every == 0:
                print('batch {}/{}, {} files processed'.format(count, len(loader), count * batch_size))

        asset_dict = {'dct_y': dct_y, 'coord': coord, 'label': label}
        save_hdf5(output_path, asset_dict, attr_dict=None, mode=mode)
        mode = 'a'

    return output_path

parser = argparse.ArgumentParser(description='Feature Extraction')
parser.add_argument('--data_h5_dir', type=str, default=None)
parser.add_argument('--data_slide_dir', type=str, default=None)
parser.add_argument('--csv_path', type=str, default=None)
parser.add_argument('--feat_dir', type=str, default=None)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--slide_ext', type=str, default='.svs')
parser.add_argument('--no_auto_skip', default=False, action='store_true')
parser.add_argument('--dct_process', default=False, action='store_true')
parser.add_argument('--pretrained', default=True, action='store_true')
parser.add_argument('--just_patches', default=False, action='store_true')
args = parser.parse_args()

if __name__ == '__main__':

    total_time_start = time.time()
    #构建目录并输出其中已有的pt文件，防止后续过程中出现重复处理的情况
    os.makedirs(args.feat_dir, exist_ok=True)
    os.makedirs(os.path.join(args.feat_dir, 'pt_files'), exist_ok=True)
    os.makedirs(os.path.join(args.feat_dir, 'h5_files'), exist_ok=True)
    dest_files = os.listdir(os.path.join(args.feat_dir, 'pt_files'))
    # #不进行频域处理：加载预训练res50模型用于后续提取切片特征
    if not args.dct_process:
        print('loading model checkpoint')
        model = resnet50_baseline(pretrained=True)
        model = model.to(device)
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model.eval()
    #下面是大模型版本 
#     if not args.dct_process:
#         print('loading model checkpoint')
        
#         model = timm.create_model(
#             "vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True
#         )
#         model.load_state_dict(torch.load(os.path.join("./models/uni/", "pytorch_model.bin"), map_location="cpu"), strict=True)
#         transform = transforms.Compose(
#             [
#                 transforms.Resize(224),
#                 transforms.ToTensor(),
#                 transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
#             ]
#         )
#         model = model.to(device)
#         model.eval()
        
#     if not args.dct_process:
#         print('loading model checkpoint')
        
#         model = timm.create_model(
#             "hf_hub:prov-gigapath/prov-gigapath",
#             pretrained=True
#         )
        
#         transform = transforms.Compose(
#             [
#                 transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
#                 transforms.CenterCrop(224),
#                 transforms.ToTensor(),
#                 transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
#             ]
#         )
#         model = model.to(device)
#         model.eval()

    # if not args.dct_process:
    #     print('loading model checkpoint')
    #     timm_kwargs = {
    #                 'model_name': 'vit_giant_patch14_224',
    #                 'img_size': 224, 
    #                 'patch_size': 14, 
    #                 'depth': 24,
    #                 'num_heads': 24,
    #                 'init_values': 1e-5, 
    #                 'embed_dim': 1536,
    #                 'mlp_ratio': 2.66667*2,
    #                 'num_classes': 0, 
    #                 'no_embed_class': True,
    #                 'mlp_layer': timm.layers.SwiGLUPacked, 
    #                 'act_layer': torch.nn.SiLU, 
    #                 'reg_tokens': 8, 
    #                 'dynamic_img_size': True
    #             }
    #     model = timm.create_model(
    #         pretrained=False, **timm_kwargs
    #     )
    #     model.load_state_dict(torch.load(os.path.join("./models/uni2/", "pytorch_model.bin"), map_location="cpu"), strict=True)
    #     transform = transforms.Compose(
    #         [
    #             transforms.Resize(224),
    #             transforms.ToTensor(),
    #             transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    #         ]
    #     )
    #     model = model.to(device)
    #     model.eval()
        
    #获取待处理的WSI路径并准备输出路径
    csv_path = args.csv_path
    bags_dataset = Dataset_All_Bags(csv_path)
    total = len(bags_dataset)
    for bag_candidate_idx in range(total):
        slide_id = bags_dataset[bag_candidate_idx].split(args.slide_ext)[0]
        bag_name = slide_id + '.h5'
        h5_file_path = os.path.join(args.data_h5_dir, 'patches', bag_name)
        slide_file_path = os.path.join(args.data_slide_dir, slide_id + args.slide_ext)
        print('\nprogress: {}/{}'.format(bag_candidate_idx, total))
        print(slide_id)

        if not args.no_auto_skip and slide_id+'.pt' in dest_files:
            print('skipped {}'.format(slide_id))
            continue
            
        if not os.path.exists(h5_file_path):#没有对应坐标就跳过
            print(f"h5 file not found for {slide_id}, skipping")
            with open(os.path.join(args.feat_dir, 'missing_h5_files.txt'), 'a') as f:
                f.write(f"{slide_id}\n")
            continue

        output_path = os.path.join(args.feat_dir, 'h5_files', bag_name)
        time_start = time.time()

        try:
            wsi = openslide.open_slide(slide_file_path)
        except:
            continue

        if not args.dct_process:
            output_file_path = compute_w_loader(h5_file_path, output_path, wsi,
                                                model=model, batch_size=args.batch_size, verbose=1, print_every=20,
                                                pretrained=args.pretrained, custom_transforms=None, just_patches=args.just_patches)
            time_elapsed = time.time() - time_start
            print('\ncomputing features for {} took {} s'.format(output_file_path, time_elapsed))
            file = h5py.File(output_file_path, "r")

            features = file['features'][:]
            print('features size: ', features.shape)
            print('coordinates size: ', file['coords'].shape)
            features = torch.from_numpy(features)
            bag_base, _ = os.path.splitext(bag_name)
            torch.save(features, os.path.join(args.feat_dir, 'pt_files', bag_base + '.pt'))

        else:
            label_file_path = os.path.join('function_code/results', slide_id + '.txt')
            output_file_path = compute_w_loader_dct(h5_file_path, output_path, wsi,
                                                    batch_size=args.batch_size, verbose=1, print_every=20,
                                                    pretrained=args.pretrained, custom_transforms=None,
                                                    label_file_path = label_file_path, only_y = True)
            time_elapsed = time.time() - time_start
            print('\ncomputing features for {} took {} s'.format(output_file_path, time_elapsed))

    total_time_elapsed = time.time() - total_time_start
    print('\ntotal feature extraction took {} s'.format(total_time_elapsed))



