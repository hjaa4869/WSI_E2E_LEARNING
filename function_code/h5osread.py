import os
import h5py

def print_h5_info(directory):
    # 遍历目录中的所有文件
    for filename in os.listdir(directory):
        # 如果是 .h5 文件
        if filename.endswith('.h5'):
            file_path = os.path.join(directory, filename)
            print(f"Opening file: {filename}")
            
            # 使用 h5py 打开 h5 文件
            with h5py.File(file_path, 'r') as f:
                print(f"Contents of {filename}:")
                # 遍历文件中的所有数据集
                for key in f.keys():
                    dataset = f[key]
                    # 打印每个数据集的名字和对应的维度
                    print(f"  {key}: {dataset.shape}")

# 指定包含 h5 文件的目录路径
directory_path = 'patches/Camelyon16_patch256_ostu/patches'
print_h5_info(directory_path)
