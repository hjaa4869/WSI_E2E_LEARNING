import h5py
import numpy as np
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA



def load_single_h5(
        h5_path,
        dataset_name="feature",
        max_samples=50000
):

    with h5py.File(
        h5_path,
        "r"
    ) as f:

        features = f[dataset_name][:]


    print(
        "feature shape:",
        features.shape
    )


    if len(features) > max_samples:

        index = np.random.choice(
            len(features),
            max_samples,
            replace=False
        )

        features = features[index]


    return features



def tsne_plot(features):

    features = StandardScaler().fit_transform(
        features
    )


    # PCA
    pca = PCA(
        n_components=50,
        random_state=42
    )

    features = pca.fit_transform(
        features
    )


    tsne = TSNE(
        n_components=2,
        perplexity=30,
        n_iter=1000,
        random_state=42
    )


    result = tsne.fit_transform(
        features
    )


    plt.figure(
        figsize=(8,6)
    )

    plt.scatter(
        result[:,0],
        result[:,1],
        s=5
    )

    plt.title(
        "t-SNE visualization"
    )

    plt.show()



# =====================
# 使用
# =====================


features = load_single_h5(
    "./data_feat/panda_uni/h5_files/006f6aa35a78965c92fffd1fbd53a058.h5",
    dataset_name="features"
)


tsne_plot(features)