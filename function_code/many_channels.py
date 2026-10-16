# import cv2
# import numpy as np
# import matplotlib.pyplot as plt

# class cv2dct(object):

#     def single_dct(self, img):
#         img_data = img.astype(float)
#         m, n = img_data.shape
#         img_dct = np.zeros(img.shape)
#         img_dct_zeroed = np.zeros_like(img_dct)
#         x_batchsize = int(m / 8)
#         y_batchsize = int(n / 8)
        
#         sum_dct = np.zeros((8, 8))
#         for i in range(0, x_batchsize):
#             for j in range(0, y_batchsize):
#                 window_x_s = i * 8
#                 window_x = (i + 1) * 8
#                 window_y_s = j * 8
#                 window_y = (j + 1) * 8
#                 dct_block = cv2.dct(img_data[window_x_s:window_x, window_y_s:window_y])  # / quantization #量化主要用于压缩，暂不考虑
#                 img_dct[window_x_s:window_x, window_y_s:window_y] = dct_block
#                 sum_dct += dct_block  # 各小块分量求和

#         # 坐标列表
#                 positions = [(0,0),(0, 1), (2, 0), (1, 0), (1, 1), (0, 4), (0, 5), (3, 0), (3, 1), (2, 3)]

#                 for (x, y) in positions:
                
#             # 将指定区域的频域分量复制到新的数组中
#                     img_dct_zeroed[window_x_s:window_x, window_y_s:window_y][x, y] = img_dct[window_x_s:window_x, window_y_s:window_y][x, y]
        
#         return img_dct_zeroed
        
        
# #         # 创建一个空的列表来存储不同通道的结果
# #         channels = []

# #         # 创建一个大小为32x32的数组并填充
# #         l = len(positions)  # 这里l是positions的长度

# #         for l_index in range(l):
# #             result_array = np.zeros((32, 32))  # 初始化32x32数组,用于放置各通道频域分量的数据
# #             for a in range(32):
# #                 for b in range(32):
# #                     # 获取当前块的坐标
# #                     window_x = a * 8
# #                     window_y = b * 8
# #                     # 从img_dct中获取该位置的小块
# #                     dct_channel = img_dct[window_x:window_x + 8, window_y:window_y + 8]
# #                     # 使用指定的positions坐标来提取该小块中的对应值
# #                     pos = positions[l_index]
# #                     result_array[a, b] = dct_channel[pos[0], pos[1]]  # 赋值到result_array

# #             # 将每个生成的32x32数组作为一个新的通道，存入channels
# #             channels.append(result_array)

# #         # 将多个32x32数组合并为一个多通道的结果
# #         resize_channel = np.stack(channels, axis=-1)  # 沿着最后一个轴合并，形成多通道图像
       

# #         return resize_channel///def many_dct(self, img):这里分开，如果取消注释的话
    
#         # 确保输入是三通道图像
#         if len(img.shape) != 3 or img.shape[2] != 3:
#             raise ValueError("输入必须是三通道图像（H, W, 3）")

#         # 转换为浮点型（DCT需要浮点输入）
#         ycrcb_img = img.astype(np.float32)

#         # 分离通道（假设输入是BGR，需先转YCbCr；如果已经是YCbCr则跳过）
#         #ycrcb_img = cv2.cvtColor(img_data, cv2.COLOR_BGR2YCrCb)  # 如果是BGR输入
#         # ycrcb_img = img_data  # 如果输入已经是YCbCr

#         # 初始化三个通道的DCT结果矩阵
#         h, w, _ = img.shape
#         dct_y = np.zeros((h, w), dtype=np.float32)
#         dct_cb = np.zeros((h, w), dtype=np.float32)
#         dct_cr = np.zeros((h, w), dtype=np.float32)

#         # 分块处理（8x8）
#         x_batchsize = h // 8
#         y_batchsize = w // 8
        
#         sum_dct_y = np.zeros((8, 8))
#         sum_dct_cr = np.zeros((8, 8))
#         sum_dct_cb = np.zeros((8, 8))
        
#         for i in range(x_batchsize):
#             for j in range(y_batchsize):
#                 # 当前块的像素范围
#                 x_start, x_end = i * 8, (i + 1) * 8
#                 y_start, y_end = j * 8, (j + 1) * 8

#                 # 对Y通道做DCT
#                 block_y = ycrcb_img[x_start:x_end, y_start:y_end, 0]
#                 dct_y[x_start:x_end, y_start:y_end] = cv2.dct(block_y)
#                 sum_dct_y += cv2.dct(block_y)

#                 # 对Cb通道做DCT
#                 block_cb = ycrcb_img[x_start:x_end, y_start:y_end, 1]
#                 dct_cb[x_start:x_end, y_start:y_end] = cv2.dct(block_cb)
#                 sum_dct_cb += cv2.dct(block_cb)

#                 # 对Cr通道做DCT
#                 block_cr = ycrcb_img[x_start:x_end, y_start:y_end, 2]
#                 dct_cr[x_start:x_end, y_start:y_end] = cv2.dct(block_cr)
#                 sum_dct_cr += cv2.dct(block_cr)

#         return sum_dct_y, sum_dct_cb, sum_dct_cr
#         #return dct_y, dct_cb, dct_cr
    
# #dct = cv2dct()

# #img = cv2.imread('patch_3115_ai_y.jpg',cv2.IMREAD_GRAYSCALE)

# #dct_y = dct.single_dct(img)

# #print(dct_y.shape)


import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dct
class cv2dct(object):

    
#     def single_dct(self, img, positions):
#         img = img.astype(float)
#         m, n = img.shape
#         x_batchsize, y_batchsize = m // 8, n // 8

#         # (num_blocks, 8, 8)
#         blocks = img.reshape(x_batchsize, 8, y_batchsize, 8).swapaxes(1, 2).reshape(-1, 8, 8)

#         # 对每个 block 做 DCT（沿两个轴）
#         dct_blocks = dct(dct(blocks, axis=1, norm='ortho'), axis=2, norm='ortho')

#         # 预处理 positions
#         pos_indices = np.array([i * 8 + j for (i, j) in positions], dtype=int)

#         # 提取通道系数
#         resize_channel = np.take(dct_blocks.reshape(-1, 64), pos_indices, axis=1)

#         # reshape 回 (x_batchsize, y_batchsize, num_channels)
#         return resize_channel.reshape(x_batchsize, y_batchsize, -1)
    
    def single_dct(self, img, positions):
        img_data = img.astype(float)
        m, n = img_data.shape
        x_batchsize = m // 8  # 块的行数（如256x256图像为32）
        y_batchsize = n // 8  # 块的列数
        
        # 1. 预处理positions：将(i,j)转换为8x8块内的线性索引（i*8 + j）
        pos_indices = np.array([i*8 + j for (i,j) in positions], dtype=int)
        num_channels = len(pos_indices)  # 通道数=positions长度
        
        # 2. 初始化结果数组：(x_batchsize, y_batchsize, num_channels)
        resize_channel = np.zeros((x_batchsize, y_batchsize, num_channels), dtype=img_data.dtype)
        
        # 3. 遍历所有8x8块，用向量化操作提取多通道数据
        for i in range(x_batchsize):
            for j in range(y_batchsize):
                # 提取当前8x8块
                x_start, x_end = i*8, (i+1)*8
                y_start, y_end = j*8, (j+1)*8
                dct_block = cv2.dct(img_data[x_start:x_end, y_start:y_end])  # 8x8的DCT块
                
                # 关键：一次性提取所有positions对应的系数，形状为(num_channels,)
                # 等价于循环赋值，但用向量化操作替代
                resize_channel[i, j, :] = np.take(dct_block, pos_indices)
        
        return resize_channel
    
#     def single_dct(self, img, positions):
#         img_data = img.astype(float)
#         m, n = img_data.shape
#         img_dct = np.zeros(img.shape)
#         img_dct_zeroed = np.zeros_like(img_dct)
#         x_batchsize = int(m / 8)
#         y_batchsize = int(n / 8)
        
#         sum_dct = np.zeros((8, 8))
#         for i in range(0, x_batchsize):
#             for j in range(0, y_batchsize):
#                 window_x_s = i * 8
#                 window_x = (i + 1) * 8
#                 window_y_s = j * 8
#                 window_y = (j + 1) * 8
#                 dct_block = cv2.dct(img_data[window_x_s:window_x, window_y_s:window_y])  # / quantization
#                 img_dct[window_x_s:window_x, window_y_s:window_y] = dct_block
#                 sum_dct += dct_block  # 各小块分量求和

#         # 创建一个空的列表来存储不同通道的结果
#         channels = []

#         # 创建一个大小为32x32的数组并填充
#         l = len(positions)  # 这里l是positions的长度

#         for l_index in range(l):
#             result_array = np.zeros((32, 32))  # 初始化32x32数组,用于放置各通道频域分量的数据
#             for a in range(32):
#                 for b in range(32):
#                     # 获取当前块的坐标
#                     window_x = a * 8
#                     window_y = b * 8
#                     # 从img_dct中获取该位置的小块
#                     dct_channel = img_dct[window_x:window_x + 8, window_y:window_y + 8]
#                     # 使用指定的positions坐标来提取该小块中的对应值
#                     pos = positions[l_index]
#                     result_array[a, b] = dct_channel[pos[0], pos[1]]  # 赋值到result_array

#             # 将每个生成的32x32数组作为一个新的通道，存入channels
#             channels.append(result_array)

#         # 将多个32x32数组合并为一个多通道的结果
#         resize_channel = np.stack(channels, axis=-1)  # 沿着最后一个轴合并，形成多通道图像
#         return resize_channel

    def many_dct(self, img):
        
        positions_Y = [(0,0),(0, 1),(0,2),(0,3),(0,4),(0,5),(0,7),(1,0), (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 7), (3, 0), (3, 1), (3, 2),(3,3),(3,5),(3,6), (3, 7)]
        positions_CR = [(0, 0), (0, 1), (0,2), (0, 3), (0,4), (0, 5), (0, 6), (0, 7), (1,0), (1, 1), (1, 2), (1, 3), (1,4),(1,5), (2, 0), (4, 0)]
        positions_CB = [(0, 0), (0, 1), (0,2), (0, 3), (0,4), (0, 5), (0, 6), (0, 7), (1,0), (1, 1), (1, 2), (1, 3), (1,4),(1,5),(1,6), (3, 0)]

        # 分离图像的三个通道
        img_Y = img[:, :, 0]
        img_CR = img[:, :, 1]
        img_CB = img[:, :, 2]

        # 对每个通道分别调用single_dct
        dct_Y = self.single_dct(img_Y, positions_Y)
        dct_CR = self.single_dct(img_CR, positions_CR)
        dct_CB = self.single_dct(img_CB, positions_CB)

        # 将三个通道的结果合并成一个多通道的图像
        dct_img = np.concatenate([dct_Y, dct_CR, dct_CB], axis=-1)
        
        return dct_img
