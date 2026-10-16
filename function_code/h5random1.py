import h5py
import numpy as np
from tqdm import tqdm  # 用于显示进度条

# 设置随机种子
np.random.seed(42)

# 输入和输出文件路径
input_file = 'data_feat/dcty_h5/labels_1.h5'
output_file = 'data_feat/dcty_h5/labels_turmor.h5'

# 1. 获取数据集的总样本数
with h5py.File(input_file, 'r') as f:
    total_samples = f['dct'].shape[0]
    print(f"原始数据集包含 {total_samples:,} 个样本")

# 2. 生成随机索引
print("生成随机索引...")
indices = np.random.choice(total_samples, 100000, replace=False)
indices = np.sort(indices)  # 排序以提高读取效率

# 3. 分批次处理
chunk_size = 10000  # 根据你的内存调整
with h5py.File(input_file, 'r') as f_in, h5py.File(output_file, 'w') as f_out:
    dataset = f_in['dct']
    sample_shape = dataset.shape[1:]
    dtype = dataset.dtype
    
    # 创建输出数据集
    out_dataset = f_out.create_dataset('dct', 
                                      shape=(100000,) + sample_shape,
                                      dtype=dtype)
    
    # 使用tqdm显示进度
    print(f"正在处理 {len(indices):,} 个样本，分块大小: {chunk_size}")
    progress_bar = tqdm(total=len(indices), unit='samples', desc='处理进度')
    
    for i in range(0, len(indices), chunk_size):
        batch_indices = indices[i:i+chunk_size]
        batch_data = dataset[batch_indices]
        out_dataset[i:i+len(batch_indices)] = batch_data
        progress_bar.update(len(batch_indices))
    
    progress_bar.close()

print("处理完成！")