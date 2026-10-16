import torch

# 加载 .pt 文件
data = torch.load('data_feat/dcty_pt/turmor1.pt')  # 替换为你的文件名

# 打印原始形状确认
print("原始形状:", data.shape)  # 应该是 [4000, 32, 32, 10]

# 调整维度顺序：从 [4000, 32, 32, 10] → [4000, 10, 32, 32]
data = data.permute(0, 3, 1, 2)

# 打印新形状确认
print("调整后形状:", data.shape)  # 应该是 [4000, 10, 32, 32]

# 保存为新文件
torch.save(data, 'data_feat/dcty_pt/turmor.pt')
