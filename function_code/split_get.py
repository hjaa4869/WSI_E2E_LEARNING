import pandas as pd

df = pd.read_csv('dataset_csv/tcga_brca_fold.csv')

slide_col = df.columns[1]  # 第2列
fold_col = df.columns[3]   # 第4列

result = pd.concat(
    [
        df[df[fold_col] == 'train'][slide_col].reset_index(drop=True),
        df[df[fold_col] == 'val'][slide_col].reset_index(drop=True),
        df[df[fold_col] == 'test'][slide_col].reset_index(drop=True),
    ],
    axis=1
)

result.columns = ['train', 'val', 'test']
result.to_csv('dataset_csv/tcga_brca_split.csv', index=False)