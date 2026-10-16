import os
import random
import h5py
import numpy as np

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE



# ====================================
# 1. 从目录随机采样feature
# ====================================

def sample_features_from_directory(
        feature_dir,
        dataset_name="feature",
        num_files=50,
        samples_per_file=1000
):

    """
    从大量h5文件随机采样feature

    Parameters
    ----------
    feature_dir:
        h5文件目录

    dataset_name:
        h5里面feature的名字

    num_files:
        随机选择多少个h5

    samples_per_file:
        每个h5采样多少feature

    """

    h5_files = [
        f for f in os.listdir(feature_dir)
        if f.endswith(".h5")
    ]


    print(
        "总h5数量:",
        len(h5_files)
    )


    # 随机选择文件

    selected_files = random.sample(
        h5_files,
        min(num_files, len(h5_files))
    )


    feature_list = []


    for filename in selected_files:


        filepath = os.path.join(
            feature_dir,
            filename
        )


        with h5py.File(filepath,"r") as f:


            feature = f[dataset_name]


            total_samples = feature.shape[0]


            sample_num = min(
                samples_per_file,
                total_samples
            )


            index = np.random.choice(
                total_samples,
                sample_num,
                replace=False
            )


            sampled_feature = feature[index]


            feature_list.append(
                sampled_feature
            )


        print(
            filename,
            "sample:",
            sample_num
        )



    features = np.concatenate(
        feature_list,
        axis=0
    )


    print(
        "最终feature:",
        features.shape
    )


    return features



# ====================================
# 2. t-SNE
# ====================================

def run_tsne(
        features,
        save_path="tsne_result.png"
):


    print(
        "开始标准化..."
    )


    features = StandardScaler().fit_transform(
        features
    )


    print(
        "PCA降维..."
    )


    # 512维 -> 50维

    pca = PCA(
        n_components=50,
        random_state=42
    )


    features = pca.fit_transform(
        features
    )


    print(
        "开始t-SNE..."
    )


    tsne = TSNE(
        n_components=2,
        perplexity=30,
        learning_rate=200,
        n_iter=1000,
        random_state=42
    )


    embedding = tsne.fit_transform(
        features
    )


    print(
        "绘图..."
    )


    plt.figure(
        figsize=(8,8)
    )


    plt.scatter(
        embedding[:,0],
        embedding[:,1],
        s=5
    )


    plt.title(
        "t-SNE Visualization"
    )


    plt.xlabel(
        "t-SNE-1"
    )

    plt.ylabel(
        "t-SNE-2"
    )


    plt.tight_layout()


    plt.savefig(
        save_path,
        dpi=300
    )


    plt.show()


    print(
        "保存:",
        save_path
    )



# ====================================
# 3. 主程序
# ====================================


if __name__ == "__main__":


    feature_dir = "./data_feat/panda_uni/h5_files"


    # 例如:
    # 50个h5
    # 每个抽1000个

    features = sample_features_from_directory(
        feature_dir,
        dataset_name="features",
        num_files=50,
        samples_per_file=1000
    )


    run_tsne(
        features,
        save_path="tsne_result.png"
    )