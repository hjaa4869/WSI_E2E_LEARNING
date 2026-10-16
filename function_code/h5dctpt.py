import h5py
import torch
import numpy as np
from tqdm import tqdm  # 进度条（可选）

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# 加载 H5 文件（仅建立连接，不立即加载数据）
h5_path = "data_feat/Camelyon16_patch256_res50/h5_files/tumor_111.h5"
batch_size = 512  # 根据 GPU 内存调整


model = RES_50()
model.relocate()
model.load_state_dict(torch.load("results/clam_cam16_res50_ostu_init_s2021/s_0_checkpoint.pt"))
model.eval()

# 存储所有输出结果的列表
outputs = []

with h5py.File(h5_path, "r") as f:
    dct_y = f["dct_y"]  # 不立即加载数据，而是建立引用
    N = dct_y.shape[0]  # 总样本数
    
    # 分批次处理
    for i in tqdm(range(0, N, batch_size)):
        # 获取当前批次数据
        batch_data = dct_y[i:i+batch_size]  # 形状 (batch_size, 10, 32, 32)
        
        # 转换为 PyTorch Tensor 并移至 GPU
        inputs = torch.from_numpy(batch_data).float().permute(2, 0, 1).to(device)
        
        # 模型推理
        with torch.no_grad():
            batch_output = model(inputs)  # 假设输出形状 (batch_size, ...)
        
        # 移回 CPU 并存储
        outputs.append(batch_output.cpu())

# 合并所有批次的输出
final_output = torch.cat(outputs, dim=0)  # 形状 (N, ...)

# 保存到 .pt 文件
torch.save(final_output, "output_results.pt")
print(f"输出已保存至 output_results.pt，形状：{final_output.shape}")