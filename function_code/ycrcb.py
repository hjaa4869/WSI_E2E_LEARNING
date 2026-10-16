import cvfunctional as F
import cv2

# 读取图像
img1 = cv2.imread('patch_3115_ai.png')
print("Image shape:", img1.shape)
# 调用转换函数
ycrcb_img = F.to_ycrcb(img1)


# 保存图像到文件
cv2.imwrite('patch_3115_ai_y.jpg',ycrcb_img[:, :, 0])
print("image saved")
img2 = cv2.imread('patch_3115_ai_y.jpg',cv2.IMREAD_GRAYSCALE)
print("Image shape:", img2.shape)