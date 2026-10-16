import h5py
import lmdb
import pickle
import numpy as np

def h5_to_lmdb(h5_path, lmdb_path):
    # 打开 h5 文件
    with h5py.File(h5_path, "r") as h5f:
        imgs = h5f["img"][:]       # shape: (N, H, W, C)
        coords = h5f["coords"][:]  # shape: (N, 2)
        label = h5f["label"][()] if "label" in h5f else None

    num_samples = len(imgs)

    # 估算需要的存储空间（比实际稍大一点）
    map_size = imgs.nbytes * 2  

    env = lmdb.open(lmdb_path, map_size=map_size)

    with env.begin(write=True) as txn:
        for i in range(num_samples):
            sample = {
                "img": imgs[i],
                "coord": coords[i],
                "label": label
            }
            key = f"{i:08d}".encode("ascii")  # 00000001, 00000002 ...
            txn.put(key, pickle.dumps(sample))

    env.close()
    print(f"✅ 转换完成: {h5_path} → {lmdb_path}, 共 {num_samples} 条记录")
    


import glob
import os

h5_files = glob.glob("data_feat/Camelyon16_patch256_res50/h5_files/*.h5") 

for h5_path in h5_files:
    lmdb_path = h5_path.replace(".h5", ".lmdb")
    h5_to_lmdb(h5_path, lmdb_path)
    os.remove(h5_path)