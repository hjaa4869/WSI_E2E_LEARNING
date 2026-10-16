import numpy as np
import matplotlib.pyplot as plt

def plot_square_heatmap_with_numbers(txt_file, save_path=None, cmap="OrRd"):
    """
    从 txt 文件读取 n^2 个数，生成 n×n 浅色调热力矩阵，并在每个格子中间写编号
    """
    # 读取数据
    data = np.loadtxt(txt_file)

    # 确保长度是 n^2
    n = int(np.sqrt(len(data)))
    if n * n != len(data):
        raise ValueError("数据长度不是完全平方数，无法转成 n×n 矩阵")

    # reshape 成矩阵
    matrix = data.reshape((n, n))

    # 绘制热力图
    plt.figure(figsize=(8, 8))
    im = plt.imshow(matrix, cmap=cmap, vmin=0, vmax=1)  # 浅色调
    plt.colorbar(im, label="Value")

    # 在每个方格中间写编号
    for i in range(n):
        for j in range(n):
            idx = i * n + j + 1  # 编号 1~n^2
            plt.text(j, i, str(idx),
                     ha='center', va='center', color='black', fontsize=8)

    plt.title(f"channel cr")
    plt.axis("off")  # 隐藏坐标轴

    # 保存或显示
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"热力矩阵已保存到 {save_path}")
    else:
        plt.show()


# 使用示例
plot_square_heatmap_with_numbers("weight.txt", save_path="heatmap_numbered.png")



# import numpy as np
# import matplotlib.pyplot as plt

# def plot_square_heatmap_with_numbers(txt_file, save_path=None, cmap="OrRd"):
#     """
#     从 txt 文件读取 n^2 个数，生成 n×n 浅色调热力矩阵，
#     格子之间有细白色间隔，并在每个格子中间写编号
#     """
#     # 读取数据
#     data = np.loadtxt(txt_file)

#     # 确保长度是 n^2
#     n = int(np.sqrt(len(data)))
#     if n * n != len(data):
#         raise ValueError("数据长度不是完全平方数，无法转成 n×n 矩阵")

#     # reshape 成矩阵
#     matrix = data.reshape((n, n))

#     # 绘制热力图
#     fig, ax = plt.subplots(figsize=(8, 8))
#     im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=1)

#     # 添加颜色条
#     cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
#     cbar.set_label("Value")

#     # 在每个方格中间写编号
#     for i in range(n):
#         for j in range(n):
#             idx = i * n + j  # 编号 1~n^2
#             ax.text(j, i, str(idx),
#                     ha='center', va='center', color='black', fontsize=24)

#     # 设置白色网格线来分隔格子
#     ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
#     ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
#     ax.grid(which="minor", color="black", linestyle='-', linewidth=2)
#     ax.tick_params(which="minor", bottom=False, left=False)  # 去掉小刻度

#     # 标题和隐藏主坐标轴刻度
#     ax.set_title("channel cb")
#     ax.set_xticks([])
#     ax.set_yticks([])

#     # 保存或显示
#     if save_path:
#         plt.savefig(save_path, dpi=300, bbox_inches="tight")
#         print(f"热力矩阵已保存到 {save_path}")
#     else:
#         plt.show()


# # 使用示例
# plot_square_heatmap_with_numbers("weight.txt", save_path="heatmap_numbered.png")
