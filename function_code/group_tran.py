#用于切片批量的YCrCb输出
import os
import cv2
import glob
from PIL import Image
import cvfunctional as F

# 定义源目录和目标目录
source_dir = "different/normal_new"
output_dir = "different/normal_new_ycrcb"

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

# 获取所有图像文件的路径（支持多种常见格式）
image_paths = glob.glob(os.path.join(source_dir, "*.jpg")) + \
              glob.glob(os.path.join(source_dir, "*.png")) + \
              glob.glob(os.path.join(source_dir, "*.jpeg"))

# 遍历所有图像并进行处理
for image_path in image_paths:
    try:
        # 打开图像文件
        img = cv2.imread(image_path)
        
        # 获取图像的文件名并创建输出路径
        file_name = os.path.basename(image_path)
        output_path = os.path.join(output_dir, file_name)
        
        ycrcb_img = F.to_ycrcb(img)
        
        cv2.imwrite(output_path,ycrcb_img)
        
        print(f"已处理并保存图像：{file_name}")
        
    except Exception as e:
        print(f"处理图像 {image_path} 时出错: {e}")
