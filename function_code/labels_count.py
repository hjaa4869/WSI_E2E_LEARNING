import h5py
import os

def calculate_labels_sum(directory):
    total_sum = 0

    # 遍历目录中的所有文件
    for filename in os.listdir(directory):
        if filename.endswith(".h5"):
            file_path = os.path.join(directory, filename)
            try:
                with h5py.File(file_path, 'r') as f:
                    # 假设字典包含键 'labels'
                    labels = f.get('labels')
                    if labels is not None:
                        total_sum += labels[:].sum()  # 累加labels的值
            except Exception as e:
                print(f"Error processing {filename}: {e}")

    return total_sum

# 示例使用
directory_path = "data_feat/Camelyon16_patch256_res50/h5_files"  # 替换为你的目录路径
total_labels_sum = calculate_labels_sum(directory_path)
print(f"总和是: {total_labels_sum}")
