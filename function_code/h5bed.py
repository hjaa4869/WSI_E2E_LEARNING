import h5py
import os
import numpy as np

# 输入目录，包含多个h5文件
input_directory = 'data_feat/Camelyon16_patch256_res50/h5_files'

# 输出的两个文件
output_file_1 = 'data_feat/dcty_h5/labels_1.h5'
output_file_0 = 'data_feat/dcty_h5/labels_0.h5'

# 初始化输出文件，如果不存在则创建
def init_output_file(file_path):
    if not os.path.exists(file_path):
        with h5py.File(file_path, 'w') as f:
            f.create_dataset('dct', shape=(0, 32, 32, 10), maxshape=(None, 32, 32, 10), dtype='float64')

init_output_file(output_file_1)
init_output_file(output_file_0)

# 遍历目录下的所有h5文件
for file_name in os.listdir(input_directory):
    if file_name.endswith('.h5'):
        file_path = os.path.join(input_directory, file_name)
        
        with h5py.File(file_path, 'r') as f:
            dct_data = f['dct_ys'][:]  # 获取dct数据
            labels_data = f['labels'][:]  # 获取labels数据s
            
            # 遍历labels并将对应的dct分量堆叠到不同的列表
            stacked_dct_1 = []
            stacked_dct_0 = []

            for i, label in enumerate(labels_data):
                if label == 1:
                    stacked_dct_1.append(dct_data[i])
                else:
                    stacked_dct_0.append(dct_data[i])
            
            # 转为 numpy 数组
            stacked_dct_1 = np.array(stacked_dct_1, dtype='float64')
            stacked_dct_0 = np.array(stacked_dct_0, dtype='float64')

            # 追加写入 output_file_1
            if len(stacked_dct_1) > 0:
                with h5py.File(output_file_1, 'a') as f1:
                    dset = f1['dct']
                    old_shape = dset.shape[0]
                    new_shape = old_shape + stacked_dct_1.shape[0]
                    dset.resize((new_shape, 32, 32, 10))
                    dset[old_shape:new_shape] = stacked_dct_1

            # 追加写入 output_file_0
            if len(stacked_dct_0) > 0:
                with h5py.File(output_file_0, 'a') as f0:
                    dset = f0['dct']
                    old_shape = dset.shape[0]
                    new_shape = old_shape + stacked_dct_0.shape[0]
                    dset.resize((new_shape, 32, 32, 10))
                    dset[old_shape:new_shape] = stacked_dct_0
