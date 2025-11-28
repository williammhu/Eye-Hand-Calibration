import cv2
import numpy as np
from openni import openni2
from openni import _openni2 as c_api
import time
import cv2.aruco as aruco

def initialize_openni(redist_path):
    openni2.initialize(redist_path)
    return openni2.Device.open_any()

def configure_depth_stream(device, width, height, fps):
    depth_stream = device.create_depth_stream()
    depth_stream.set_video_mode(c_api.OniVideoMode(
        pixelFormat=c_api.OniPixelFormat.ONI_PIXEL_FORMAT_DEPTH_1_MM,
        resolutionX=width, resolutionY=height, fps=fps
    ))
    depth_stream.start()
    return depth_stream
def convert_depth_to_xyz(u, v, depth_value, fx, fy, cx, cy):
    depth = depth_value * 0.1
    z = float(depth)
    x = float((u - cx) * z) / fx
    y = float((v - cy) * z) / fy
    return [x, -y, z]

def main():
    redist_path = "F:\study-python\dabeipro\OpenNI_2.3.0.86_202210111950_4c8f5aa4_beta6_windows\Win64-Release\sdk\libs"
    width, height, fps = 640, 400, 30
    fx, fy, cx, cy = 524.382751, 524.382751, 324.768829, 212.350189

    dev = initialize_openni(redist_path)
    print(dev.get_device_info())

    depth_stream = configure_depth_stream(dev, width, height, fps)

    cv2.namedWindow('depth')
    cv2.namedWindow('color')

    cap = cv2.VideoCapture(0)

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDBLCLK:
            print(x, y, img[y, x])
            print("三维坐标：", convert_depth_to_xyz(x, y, img[y, x], fx, fy, cx, cy))

    cv2.setMouseCallback('color', mouse_callback)

    # 创建ArUco字典，可根据检测的字典家族修改参数
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_7X7_100)

    # 设置ArUco标记的大小
    marker_size = 5

    # 创建ArUco参数
    parameters = aruco.DetectorParameters()
    fps = 0.0
    # 设置视频输出参数
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 选择编码器（这里以MP4V为例，根据需要可更换）
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 获取当前摄像头的宽度
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 获取当前摄像头的高度
    # output_file = 'utils/output1.mp4'  # 输出视频文件名

    # 创建VideoWriter对象
    # out = cv2.VideoWriter(output_file, fourcc, 30, (frame_width, frame_height))

    while True:
        t1 = time.time()
        frame = depth_stream.read_frame()
        frame_data = frame.get_buffer_as_uint16()
        img = np.ndarray((frame.height, frame.width), dtype=np.uint16, buffer=frame_data)
        dim_gray = cv2.convertScaleAbs(img, alpha=0.17)

        kernel_size = 5
        dim_gray = cv2.medianBlur(dim_gray, kernel_size)

        depth_colormap = cv2.applyColorMap(dim_gray, 2)

        ret, frame_color = cap.read()
        frame_color = cv2.resize(frame_color, (640, 400))
        frame_color = cv2.flip(frame_color, 1)
        # out.write(frame_color)

        # 将彩色图像转换为灰度图像
        gray = cv2.cvtColor(frame_color, cv2.COLOR_BGR2GRAY)

        # 检测ArUco标识物
        detector = aruco.ArucoDetector(aruco_dict, parameters)
        # corners, ids, rejected = detector.detectMarkers(gray, aruco_dict, parameters=parameters)
        corners, ids, rejected = detector.detectMarkers(gray )

        if len(corners) > 0:
            # 绘制检测到的ArUco标识物的边界框
            # print(ids)
            aruco.drawDetectedMarkers(frame_color, corners, ids)


        fps = (fps + (1. / (time.time() - t1))) / 2
        depth_colormap = cv2.putText(depth_colormap, "fps= %.2f" % (fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                     (0, 255, 0), 2)
        cv2.imshow('color', frame_color)
        cv2.imshow('depth', depth_colormap)

        key = cv2.waitKey(10)
        if int(key) == "q":
            break

    openni2.unload()
    dev.close()


if __name__ == "__main__":
    main()
