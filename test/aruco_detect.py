# import imutils
import cv2
import sys
from cv2 import aruco


# 定义 OpenCV 支持的每个可能的 ArUco 标签的名称
ARUCO_DICT = {"DICT_4X4_50": cv2.aruco.DICT_4X4_50, "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
              "DICT_4X4_250": cv2.aruco.DICT_4X4_250, "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
              "DICT_5X5_50": cv2.aruco.DICT_5X5_50, "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
              "DICT_5X5_250": cv2.aruco.DICT_5X5_250, "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
              "DICT_6X6_50": cv2.aruco.DICT_6X6_50, "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
              "DICT_6X6_250": cv2.aruco.DICT_6X6_250, "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
              "DICT_7X7_50": cv2.aruco.DICT_7X7_50, "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
              "DICT_7X7_250": cv2.aruco.DICT_7X7_250, "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
              "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
              "DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
              "DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
              "DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
              "DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11}

image = cv2.imread('test_aruco5-5-50.jpg')
image = cv2.resize(image, (640, 400))
corners = []
ids = []
rejected = []
for key, value in ARUCO_DICT.items():
    arucoDict = cv2.aruco.getPredefinedDictionary(value)
    arucoParams = cv2.aruco.DetectorParameters()
    detector = aruco.ArucoDetector(arucoDict, arucoParams)
    corners, ids, rejected = detector.detectMarkers(image, )
    if len(corners) > 0:
        print(key)
        print(corners)
        break

# 验证至少一个 ArUCo 标记被检测到
if len(corners) > 0:
    # 展平 ArUCo ID 列表
    ids = ids.flatten()
    # 循环检测到的 ArUCo 标记
    for (markerCorner, markerID) in zip(corners, ids):
        # 提取始终按​​以下顺序返回的标记：
        # TOP-LEFT, TOP-RIGHT, BOTTOM-RIGHT, BOTTOM-LEFT
        corners = markerCorner.reshape((4, 2))
        (topLeft, topRight, bottomRight, bottomLeft) = corners
        # 将每个 (x, y) 坐标对转换为整数
        topRight = (int(topRight[0]), int(topRight[1]))
        bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
        bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
        topLeft = (int(topLeft[0]), int(topLeft[1]))
        # 绘制ArUCo检测的边界框
        cv2.line(image, topLeft, topRight, (0, 255, 0), 2)
        cv2.line(image, topRight, bottomRight, (0, 255, 0), 2)
        cv2.line(image, bottomRight, bottomLeft, (0, 255, 0), 2)
        cv2.line(image, bottomLeft, topLeft, (0, 255, 0), 2)
        # 计算并绘制 ArUCo 标记的中心 (x, y) 坐标
        cX = int((topLeft[0] + bottomRight[0]) / 2.0)
        cY = int((topLeft[1] + bottomRight[1]) / 2.0)
        cv2.circle(image, (cX, cY), 4, (0, 0, 255), -1)
        # 在图像上绘制 ArUco 标记 ID
        cv2.putText(image, str(markerID), (topLeft[0], topLeft[1] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        print("[INFO] ArUco marker ID: {}".format(markerID))
        # 显示输出图像
cv2.namedWindow("Image1", 0);
cv2.resizeWindow("Image1", 640, 480);
cv2.imshow("Image1", image)
cv2.waitKey(0)


# import cv2
# import cv2.aruco as aruco
# import numpy as np
#
# from orbbec_init import initialize_openni, configure_depth_stream, convert_depth_to_xyz
#
# redist_path = "F:\study-python\dabeipro\OpenNI_2.3.0.86_202210111950_4c8f5aa4_beta6_windows\Win64-Release\sdk\libs"
# width, height, fps = 640, 400, 30
# # fx, fy, cx, cy = 524.382751, 524.382751, 324.768829, 212.350189
# #
# # def mouse_callback(event, x, y, flags, param):
# #     if event == cv2.EVENT_LBUTTONDBLCLK:
# #         print(x, y, frame[y, x])
# #         print("三维坐标：", convert_depth_to_xyz(x, y, frame[y, x], fx, fy, cx, cy))
#
# dev = initialize_openni(redist_path)
# print(dev.get_device_info())
# # cv2.setMouseCallback('ArUco Detection', mouse_callback)
# cap = cv2.VideoCapture(0)
# cap.set(3, 640)  # 设置视频宽度为640像素
# cap.set(4, 400)  # 设置视频高度为480像素
#
# # 创建ArUco字典，可根据检测的字典家族修改参数
# aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_5X5_100)
#
# # 设置ArUco标记的大小
# marker_size = 5
#
# # 创建ArUco参数
# parameters = aruco.DetectorParameters()
#
#
#
# while True:
#
#     ret, frame = cap.read()
#     frame = cv2.resize(frame, (640, 400), interpolation=cv2.INTER_CUBIC)
#     # frame = frame[0:0, 640:400]
#     # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     flipped = cv2.flip(frame, 1)
#
#     # 检测ArUco标识物
#     corners, ids, rejected = aruco.detectMarkers(flipped, aruco_dict, parameters=parameters)
#
#     if len(corners) > 0:
#         # 绘制检测到的ArUco标识物的边界框
#         print(ids)
#         aruco.drawDetectedMarkers(flipped, corners, ids)
#
#     # 显示图像
#     cv2.imshow('ArUco Detection', flipped)
#
#     # 按下ESC键退出程序
#     if cv2.waitKey(1) == 27:
#         break
#
# # 释放摄像头对象和关闭窗口
# cap.release()
# cv2.destroyAllWindows()
