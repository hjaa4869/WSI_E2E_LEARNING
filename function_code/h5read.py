import h5py
import torch
import numpy as np

def read_h5_file(file_path):
    try:
        with h5py.File(file_path, 'r') as file:
            print("HDF5 文件中的数据集信息：")
            for key in file.keys():
                dataset = file[key]
                data = dataset[:]
                print(f"键: {key}")
                print(f"数据类型: {dataset.dtype}")
                print(f"维度: {dataset.shape}")
                print(f"数据内容（部分）: {data.flatten()[:30]}")

                if isinstance(data, np.ndarray):
                    print("数据类型为 numpy 数组")
                elif isinstance(data, torch.Tensor):
                    print("数据类型为 PyTorch 张量")
                else:
                    print("未知的数据类型")

                print("-" * 50)
    except FileNotFoundError:
        print("错误: 文件未找到!")
    except Exception as e:
        print(f"错误: 发生了一个未知错误: {e}")



if __name__ == "__main__":
    file_path = './data_feat/panda_uni/h5_files/0005f7aaab2800f6170c399693a96917.h5'#大文件，谨慎运行
    read_h5_file(file_path)
