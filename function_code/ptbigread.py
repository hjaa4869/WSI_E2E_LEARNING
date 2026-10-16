import os
import torch
from typing import Dict, Any

def load_pt_files(folder_path: str) -> Dict[str, Any]:
    """
    读取文件夹中的所有 .pt 文件，并返回一个字典，键是文件名，值是加载的数据。

    Args:
        folder_path (str): 包含 .pt 文件的文件夹路径

    Returns:
        Dict[str, Any]: 文件名到数据的映射
    """
    pt_files = {}
    
    # 确保文件夹存在
    if not os.path.isdir(folder_path):
        raise ValueError(f"文件夹不存在: {folder_path}")
    
    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        if filename.endswith('.pt'):
            file_path = os.path.join(folder_path, filename)
            try:
                # 加载 .pt 文件
                data = torch.load(file_path)
                pt_files[filename] = data
                print(f"成功加载: {filename}")
            except Exception as e:
                print(f"加载 {filename} 失败: {e}")
    
    return pt_files

# 示例用法
if __name__ == "__main__":
    folder_path = "data_feat/Camelyon16_patch256_res50/pt_files"  # 替换为你的文件夹路径
    pt_data = load_pt_files(folder_path)
    
    # 打印加载的文件信息
    for name, data in pt_data.items():
        print(f"\n文件名: {name}")
        print(f"数据类型: {type(data)}")
        
        # 如果是张量，打印形状
        if isinstance(data, torch.Tensor):
            print(f"张量形状: {data.shape}")
        # 如果是模型，打印模型结构
        elif isinstance(data, torch.nn.Module):
            print(f"模型结构: {data}")