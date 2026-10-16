import h5py
import os
import numpy as np

def stack_features_from_directory(input_directory, output_file):
    all_features = []  # 用来存储所有的feature数据
    
    # 遍历目录中的所有h5文件
    for filename in os.listdir(input_directory):
        if filename.endswith(".h5"):
            file_path = os.path.join(input_directory, filename)
            try:
                with h5py.File(file_path, 'r') as f:
                    
                    feature = f['dct_ys'][:]
                    if feature is not None:
                      
                        all_features.append(feature)  # 将数据添加到列表中
            except Exception as e:
                print(f"Error processing {filename}: {e}")
    # 将所有feature堆叠成一个大数组
    stacked_features = np.vstack(all_features)  # 按行堆叠所有数据
    # 保存到新的h5文件
    with h5py.File(output_file, 'w') as f:
        f.create_dataset('dct_ys', data=stacked_features)
        print(f"堆叠后的数据已保存到 {output_file}")

# 示例使用
input_directory = "data_feat/Camelyon16_patch256_res50/h5_files"  # 替换为你的输入目录路径
output_file = "data_feat/dcty_h5/normal_dct.h5"  # 输出的h5文件路径
stack_features_from_directory(input_directory, output_file)
