import torch
import torch.nn as nn
from math import floor
import os
import random
import numpy as np
import pdb
import time
from PIL import Image
import h5py
import openslide


file = h5py.File('data_feat/Camelyon16_patch256_res50/h5_files/test_035.h5', "r")

features = file['dct_ys'][:]
print('features size: ', features.shape)
features = torch.from_numpy(features)
torch.save(features, 'data_feat/dcty_pt/turmor1.pt')