import openslide
import numpy as np
from IPython.display import display
import cv2

def display_wsi_info_and_level(wsi_path, level):
    # 打开 WSI 文件
    slide = openslide.open_slide(wsi_path)

    # 获取 WSI 的尺寸
    width, height = slide.dimensions
    print("WSI尺寸：{}x{}".format(width, height))

    # 获取 WSI 的 level 数量
    num_levels = slide.level_count
    dimensions = slide.level_dimensions
    print("WSI Level数量：", num_levels)
    print("WSI Level明细：", dimensions)
    
    #获取WSI元数据，读取相关信息
    if 'aperio.AppMag' in slide.properties.keys():
        level_0_magnification = int(slide.properties['aperio.AppMag'])
    elif 'openslide.mpp-x' in slide.properties.keys():
        level_0_magnification = 40 if int(np.floor(float(slide.properties['openslide.mpp-x']) * 10)) == 2 else 20
    else:
        level_0_magnification = 40
        
    #输出level0对应的放大倍数
    print("level_0对应的放大倍数为:",level_0_magnification)
    
    magnification = 5
    # 根据levle0放大倍数和指定需要的放大倍数计算放大倍率，比如level0的放大倍数如果是40倍，那么想要5倍的话，下采样率就是40/5=8
    downsample = level_0_magnification/5

    # 获取与该下采样率对应的放大倍数最接近的 Level级别
    level = slide.get_best_level_for_downsample(downsample)

    # 打印 Level
    print("{}倍放大倍率对应Level_{}".format(magnification,level))
    

    # 显示出对应 level 的图像数据
    image = slide.read_region((0, 0), slide.level_count - 5, slide.level_dimensions[5])

    # 调用电脑相册显示图像
    image.show()
    
    # 直接在notebook上显示图像
    display(image)

    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGBA2BGR)  # 转换为 OpenCV 格式的图像
    # 保存图像为普通的图像文件
    output_path = 'path_to_output_image.jpg'  # 替换为输出图像的路径
    cv2.imwrite(output_path, image)

# 指定 WSI 文件路径和选择的 level
wsi_path = 'data_storage/panda/ffe9bcababc858e04840669e788065a1.tiff'
selected_level = 2

# 调用函数进行显示
display_wsi_info_and_level(wsi_path, selected_level)