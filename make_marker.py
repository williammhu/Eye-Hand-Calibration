import cv2
import numpy as np

# 选择与代码中一致的字典：DICT_5X5_100
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)

marker_id = 0          # 用 0 号 ID 就行，或者换成别的整数
marker_size = 400      # 输出图像的边长，单位：像素（打印时越大越清楚）

marker_img = cv2.aruco.generateImageMarker(aruco_dict, marker_id, marker_size)

# 保存为 PNG
cv2.imwrite("aruco_5x5_100_id0.png", marker_img)

print("Saved aruco_5x5_100_id0.png")
