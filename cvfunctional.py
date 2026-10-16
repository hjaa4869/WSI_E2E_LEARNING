#该文件定义一些底层的功能代码
import cv2

def to_ycrcb(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    return img





