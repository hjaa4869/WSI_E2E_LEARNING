import lmdb
import pickle
import numpy as np

# LMDB 文件夹路径
lmdb_path = "data_feat/Camelyon16_patch256_res50/h5_files/normal_001.lmdb"  # 修改为你的 LMDB 目录

# 打开 LMDB
env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)

with env.begin(write=False) as txn:
    cursor = txn.cursor()
    
    print("Keys in LMDB:")
    for key, _ in cursor:
        print(key.decode("ascii"))
    
    # 查看前几个样本的内容
    print("\nSample content:")
    count = 0
    for key, value in txn.cursor():
        sample = pickle.loads(value)
        print(f"\nKey: {key.decode('ascii')}")
        if "img" in sample:
            print("Image shape:", np.array(sample["img"]).shape)
        if "coord" in sample:
            print("Coord shape:", np.array(sample["coord"]).shape)
        if "label" in sample:
            print("Label:", sample["label"])
        
        count += 1
        if count >= 5:  # 只看前5个
            break

env.close()
