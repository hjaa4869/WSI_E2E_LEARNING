import pandas as pd
import numpy as np

# 读取CSV文件
df = pd.read_csv('10w.csv')

# 指定要打乱的列名
column_to_shuffle = 'val'

# 打乱该列的数据
shuffled_column = df[column_to_shuffle].sample(frac=1, random_state=42).reset_index(drop=True)

# 将打乱后的列替换原始列
df[column_to_shuffle] = shuffled_column

# 保存结果
df.to_csv('shuffled_data.csv', index=False)

print(f"列 '{column_to_shuffle}' 已打乱并保存为 'shuffled_data.csv'")
