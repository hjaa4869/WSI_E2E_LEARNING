import h5py
import numpy as np

with h5py.File('data_feat/Camelyon16_patch256_res50/h5_files/tumor_111.h5', 'r') as f:
    if 'coords' in f:
        coords = f['coords'][:]
    else:
        coords = f['dct'][:]  # fallback

    if 'labels' in f:
        labels = f['labels'][:]

        # 找出 labels 为 0 的索引
        indices_label_0 = np.where(labels == 1)[0]

        # 取出对应的 coords
        coords_label_0 = coords[indices_label_0]
        np.savetxt('coords_label0.txt', coords_label_0, fmt='%d', delimiter=' ')

        # 打印结果
        print("标签为 0 的 coords 信息：")
        print(f"总数量: {len(coords_label_0)}")
        print(f"示例（前5个）:\n{coords_label_0[:]}")
    else:
        print('H5 文件中没有 labels 数据')
