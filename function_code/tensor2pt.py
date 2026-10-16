import torch

random_tensor = torch.randn(150, 3, 256, 256)


try:
    torch.save(random_tensor, 'data_feat/Camelyon16_patch256_res50/pt_files/normal_001.pt')
    torch.save(random_tensor, 'data_feat/Camelyon16_patch256_res50/pt_files/tumor_001.pt')
    print("张量已成功保存到 random_tensor.pt 文件中。")
except Exception as e:
    print(f"保存文件时出现错误: {e}")   