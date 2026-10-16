import pandas as pd
import numpy as np

# 读取CSV文件
file_path = '10w.csv'  # 替换为你的CSV文件路径
df = pd.read_csv(file_path)

# # 选择要打乱顺序的列（例如列名为'column_name'）
# column_to_shuffle = 'train'  # 替换为你要打乱的列名

# # 打乱该列的顺序
# shuffled_values = df[column_to_shuffle].values
# np.random.shuffle(shuffled_values)
# df[column_to_shuffle] = shuffled_values

# 选择要打乱顺序的列（例如列名为'column_name'）
column_to_shuffle1 = 'val'  # 替换为你要打乱的列名

# 打乱该列的顺序
shuffled_values1 = df[column_to_shuffle1].values
np.random.shuffle(shuffled_values1)
df[column_to_shuffle1] = shuffled_values1

# 选择要打乱顺序的列（例如列名为'column_name'）
column_to_shuffle2 = 'test'  # 替换为你要打乱的列名

# 打乱该列的顺序
shuffled_values2 = df[column_to_shuffle2].values
np.random.shuffle(shuffled_values2)
df[column_to_shuffle2] = shuffled_values2

# 保存打乱后的数据到新的CSV文件
output_file_path = '20w_new.csv'
df.to_csv(output_file_path, index=False)

print(f"列 '{column_to_shuffle}' 已打乱顺序并保存到 {output_file_path}")