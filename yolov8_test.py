import ctypes
import os
import threading
import time

import cv2
import numpy as np
import serial
from openni import openni2
from ultralytics import YOLO

from multiprocessing import Process, Value, Queue, Array

from orbbec_init import initialize_openni, configure_depth_stream, convert_depth_to_xyz
from port_test import Ser


def orbbec_video(center_p_queue, robot_status):
    """
    使用相机进行视频捕捉和物体检测的函数。
    
    参数:
    - center_p_queue: 一个队列，用于存储检测到的物体中心点在相机坐标系下的坐标。
    - robot_status: 一个共享变量，用于指示机器人的工作状态（搜索或运行）。
    
    无返回值。
    """
    model = YOLO('best.pt')
    redist_path = "F:\study-python\dabeipro\OpenNI_2.3.0.86_202210111950_4c8f5aa4_beta6_windows\Win64-Release\sdk\libs"
    width, height, fps = 640, 400, 30
    fx, fy, cx, cy = 524.382751, 524.382751, 324.768829, 212.350189

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDBLCLK:
            print(x, y, img[y, x])
            print("三维坐标：", convert_depth_to_xyz(x, y, img[y, x], fx, fy, cx, cy))

    dev = initialize_openni(redist_path)
    print(dev.get_device_info())
    depth_stream = configure_depth_stream(dev, width, height, fps)
    cv2.namedWindow('depth')
    cv2.namedWindow('color')
    # cv2.setMouseCallback('depth', mouse_callback)
    cap = cv2.VideoCapture(0)
    fps = 0.0

    try:

        while True:
            t1 = time.time()

            # 开始读取深度视频流
            frame = depth_stream.read_frame()
            # intr = frame.profile.as_video_stream_profile().intrinsics
            # print(intr)
            frame_data = frame.get_buffer_as_uint16()
            # print(frame_data)
            img = np.ndarray((frame.height, frame.width), dtype=np.uint16, buffer=frame_data)
            dim_gray = cv2.convertScaleAbs(img, alpha=0.17)

            kernel_size = 5
            dim_gray = cv2.medianBlur(dim_gray, kernel_size)
            depth_colormap = cv2.applyColorMap(dim_gray, 2)
            # 开始读取彩色视频流
            ret, frame = cap.read()
            frame = cv2.resize(frame, (640, 400))
            frame = cv2.flip(frame, 1)
            # 开始推理
            results = model.predict(source=frame, **{'save': False, 'conf': 0.62, 'verbose': False}, )
            result = results[0].boxes.data.tolist()
            result_list = []
            max_score_bbox = [0, 0, 0, 0]
            category_dict = {
                'plastic bottle': 0,
                'glass bottle': 0,
                'mask': 1,
                'gauze': 1,
                'injector': 2
            }
            for idx in range(len(result)):
                xmin = int(result[idx][0])
                ymin = int(result[idx][1])
                xmax = int(result[idx][2])
                ymax = int(result[idx][3])
                conf = round(float(result[idx][4]), 2)
                cls_idx = int(result[idx][5])
                cls_name = model.names[cls_idx]
                result_list.append([ymin, xmin, ymax, xmax, conf, cls_idx, cls_name])
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                box_color = (0, 0, 255)

                x = int((xmax - xmin) / 2 + xmin)
                y = int((ymax - ymin) / 2 + ymin)
                # print(result_list, "中点:", x, y, conf, cls_name)
                if y > 400:
                    print("目标超出深度图像范围")
                    continue
                if conf > max_score_bbox[0]:
                    max_score_bbox[0] = conf
                    max_score_bbox[1:] = [x, y, category_dict[cls_name]]
                    # put text under box

                # center_p_queue.put([x_cam, y_cam, z_cam])
                # print("center_p_queue执行")
                x_cam, y_cam, z_cam = convert_depth_to_xyz(x, y, img[y, x], fx, fy, cx, cy)
                cv2.circle(frame, (x,y), 3, (0, 0, 255), -1)
                cv2.putText(frame, cls_name, (xmin, ymin + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, box_color, 2)
                cv2.putText(frame, "x: {:.3f}".format(x_cam), (xmin, ymin + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            box_color, 2)
                cv2.putText(frame, "y: {:.3f}".format(y_cam), (xmin, ymin + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            box_color, 2)
                cv2.putText(frame, "z: {:.3f}".format(z_cam), (xmin, ymin + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            box_color, 2)
            # print("conf最大", max_score_bbox)
            x_cam, y_cam, z_cam = convert_depth_to_xyz(max_score_bbox[1], max_score_bbox[2],
                                                       img[max_score_bbox[2], max_score_bbox[1]], fx, fy, cx, cy)

            if robot_status.value == 0 and max_score_bbox[0] > 0.25:
                center_p_queue.put([x_cam, y_cam, z_cam, max_score_bbox[-1]])
                print(x_cam, y_cam, z_cam)
                print("center_p_queue执行")
                robot_status.value = 1
            #     # 处理深度图像，生成深度图的彩色版本
            #     depth_img = np.asanyarray(depth.get_data())
            #     depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_img, alpha=alpha_val), cv2.COLORMAP_JET)
            fps = (fps + (1. / (time.time() - t1))) / 2
            depth_colormap = cv2.putText(depth_colormap, "fps= %.2f" % (fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 255, 0), 2)
            # 根据机器人的状态，在图像上显示状态信息
            if robot_status.value == 0:
                status_text = "Status: Searching"
            else:
                status_text = "Status: Running"
            cv2.putText(frame, status_text, (400, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0) if robot_status == 0 else (0, 0, 255), 2)

            # 将颜色图像和深度图像叠加显示
            # images = np.vstack((color_image, depth_colormap))
            cv2.imshow('color', frame)
            cv2.imshow('depth', depth_colormap)

            key = cv2.waitKey(10)
            if int(key) == 113:
                break
    finally:
        # pipeline.stop()  # 关闭相机和视频写入器
        openni2.unload()
        dev.close()

def send_message(contunts):
    if ser.isOpen():  # 如果串口已经打开
        if contunts:
            # com1端口向com2端口写入数据 字符串必须译码
            # self.contunts = input("输入内容:")
            # 可以自定义输入Gcode命令，规划路径，例如 G1 X10 Y10 Z10 F30，其实就是三维坐标和速度
            ser.write((contunts + "\r").encode("utf-8"))
            time.sleep(1)
            # encode()函数是将字符串转化成相应编码方式的字节形式
            # 如str2.encode('utf-8')，表示将unicode编码的字符串str2转换成utf-8编码的字节数据。
            # 如果不转换，COM1发送到COM2的信息，COM2（调试助手）中文会识别不出来或者会出现乱码现象
    else:
        print("open failed")
def robot_grasp(center_p_queue, robot_status):
    """
    使用dobot机械臂进行抓取操作。

    参数:
    - center_p_queue: 队列，包含目标中心点的位置信息。
    - robot_status: 共享变量，用于控制机械臂的状态。
    """
    if os.path.exists("./save_parms/image_to_arm.npy"):
        image_to_arm = np.load("./save_parms/image_to_arm.npy")
    else:
        print("image_to_arm.npy not exist")
        return

    # 初始化
    ser.write("G28\r".encode("utf-8"))
    # send_message("M114")
    print("发送成功")
    # robot_status.value = 0

    while True:
        # 循环等待并处理目标中心点信息
        if robot_status.value:

            center_p = center_p_queue.get()
            center = center_p[0:3]

            # 计算机械臂需要移动到的位置
            img_pos = np.ones(4)
            img_pos[0:3] = center
            arm_pos = np.dot(image_to_arm, np.array(img_pos))
            print("arm_pos", arm_pos)
            # 如果目标位置超出机械臂可达范围，则跳过此次操作

            send_message(f"G1 X{0} Y{185} Z{160}")  # 回原点

            send_message(f"G1 X{arm_pos[0]} Y{arm_pos[1]} Z{arm_pos[2]+80} ")  # 移动到目标上方50mm处

            send_message(f"G1 X{arm_pos[0]} Y{arm_pos[1]} Z{arm_pos[2]-25} ")  # 移动到目标上方处

            send_message("M5")  # 执行抓取
            time.sleep(5)
            send_message(f"G1 X{arm_pos[0]} Y{arm_pos[1]} Z{arm_pos[2]+80} ")  # 再次移动到目标上方20mm处
            if center_p[3] == 0:
                send_message(f"G1 X{170} Y{0} Z{160}")  # 病理性废物位置
            if center_p[3] == 2:
                send_message(f"G1 X{-170} Y{30} Z{160}")  # 损伤性废物位置
            if center_p[3] == 1:
                send_message(f"G1 X{170} Y{120} Z{160}")  # 感染性废物
            send_message("M3")  # 打开夹爪
            time.sleep(4)

            # action.send_message(f"G1 X{0} Y{185} Z{160} ")  # 返回原点
            print("another one")
            # 重置为搜索状态
            robot_status.value = 0

        else:
            # 如果没有检测到目标标记，重置为搜索状态
            print("no marker detected")
            time.sleep(3)
            # robot_status.value = 0
            continue


if __name__ == "__main__":
    center_arr = Array(ctypes.c_double, [0, 0, 0])  # 存储中心点坐标
    robot_status = Value(ctypes.c_int8, 0)
    center_p_queue = Queue()

    ser = serial.Serial("COM5", baudrate=115200, timeout=2, )
    time.sleep(2)

    process1 = Process(target=orbbec_video, args=(center_p_queue, robot_status,))
    # process2 = Process(target=robot_grasp, args=(center_p_queue, robot_status,))
    process1.start()
    # process2.start()
    robot_grasp(center_p_queue, robot_status)
    process1.join()
    # process2.join()
