import torch

# 加载 .pt 文件
data = torch.load('data_feat/Camelyon16_patch256_res50/pt_files/normal_092.pt')

# 如果文件是一个字典（通常用于保存多个数据，如模型、优化器状态等）
if isinstance(data, dict):
    for key, value in data.items():
        print(f"Key: {key}, Shape: {value.shape if isinstance(value, torch.Tensor) else 'Not a Tensor'}")
else:
    print(f"Shape: {data.shape if isinstance(data, torch.Tensor) else 'Not a Tensor'}")
    
