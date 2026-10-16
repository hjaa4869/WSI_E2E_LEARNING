import h5py
import numpy as np

# 读取原始 h5 文件
with h5py.File('data_feat/dcty_h5/labels_0.h5', 'r') as f:
    data = f['dct'][:]  # 替换成你真实的数据键名

# 随机选择 4000 个样本
np.random.seed(42)  # 为了可复现
indices = np.random.choice(data.shape[0], 100000, replace=False)
selected_data = data[indices]

# 保存到新的 h5 文件
with h5py.File('data_feat/dcty_h5/labels_normal.h5', 'w') as f_out:
    f_out.create_dataset('dct', data=selected_data)
