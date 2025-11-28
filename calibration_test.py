import threading

import numpy as np
import cv2
import cv2.aruco as aruco

import os
import time

import serial

from orbbec_init import initialize_openni, configure_depth_stream
from port_test import Ser


# from port_test import Ser


def convert_depth_to_xyz(u, v, depth_value, fx, fy, cx, cy):
    depth = depth_value
    z = float(depth)
    x = float((u - cx) * z) / fx
    y = float((v - cy) * z) / fy
    return x, -y, z


class Calibration:
    def __init__(self):

        # 初始化相机
        redist_path = "F:\study-python\dabeipro\OpenNI_2.3.0.86_202210111950_4c8f5aa4_beta6_windows\Win64-Release\sdk\libs"
        width, height, fps = 640, 400, 30
        self.fx, self.fy, self.cx, self.cy = 524.382751, 524.382751, 324.768829, 212.350189

        # def mouse_callback(event, x, y, flags, param):
        #     if event == cv2.EVENT_LBUTTONDBLCLK:
        #         print(x, y, img[y, x])
        #         print("三维坐标：", convert_depth_to_xyz(x, y, img[y, x], fx, fy, cx, cy))

        dev = initialize_openni(redist_path)
        print(dev.get_device_info())
        self.depth_stream = configure_depth_stream(dev, width, height, fps)

        # cv2.namedWindow('depth')
        # cv2.setMouseCallback('depth', mouse_callback)

        self.cap = cv2.VideoCapture(0)

        # 初始化用于ArUco标记的字典
        self.dictionary = aruco.getPredefinedDictionary(aruco.DICT_7X7_100)
        # 设置ArUco标记的大小
        self.marker_size = 5
        # 创建参数对象
        # self.parameters = aruco.DetectorParameters_create()
        self.parameters = aruco.DetectorParameters()

    def get_aruco_center(self, calib=True):
        """
        获取Aruco标记中心的函数。
        
        ：
        - 轮廓检测以找到Aruco标记；
        - 根据相机内参和外参估计标记的位置；
        - 可选地，进行校准过程以获取更精确的标记中心。
        
        参数:
        - calib: 布尔值，指示是否执行校准流程以获取Aruco标记的中心。默认为True。
        
        返回值:
        - images: 包含颜色图像和深度图像的堆叠图像。
        - center: 如果找到标记且calib为True，则返回标记在相机坐标系中的3D坐标。
        """
        # 创建管道对象并处理帧
        # frames = self.pipeline.wait_for_frames()
        # frames = self.align.process(frames)
        
        # 获取深度帧和颜色帧
        # depth = frames.get_depth_frame()
        # color_frame = frames.get_color_frame()
        # color_image = np.asanyarray(color_frame.get_data())

        frame = self.depth_stream.read_frame()
        frame_data = frame.get_buffer_as_uint16()
        # print(frame_data)
        img = np.ndarray((frame.height, frame.width), dtype=np.uint16, buffer=frame_data)
        dim_gray = cv2.convertScaleAbs(img, alpha=0.17)

        kernel_size = 5  # 中值滤波，滤波器的大小
        dim_gray = cv2.medianBlur(dim_gray, kernel_size)
        depth = cv2.applyColorMap(dim_gray, 2)
        # 开始读取彩色视频流
        ret, frame = self.cap.read()
        color_image = cv2.resize(frame, (640, 400))
        color_image = cv2.flip(color_image, 1)
        # gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)  # 转化为灰度图像
        
        # 在颜色图像上显示状态文本
        cv2.putText(color_image, "Status: Calibrating", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 获取相机内参
        '''
        intr = color_frame.profile.as_video_stream_profile().intrinsics
        intr_matrix = np.array([[intr.fx, 0, intr.ppx],
                                [0, intr.fy, intr.ppy],
                                [0, 0, 1]])
        intr_coeffs = np.array(intr.coeffs)  # 畸变系数
        '''
        intr_matrix = np.array([[self.fx, 0, self.cx],
                                [0, self.fy, self.cy],
                                [0, 0, 1]])
        # intr_coeffs = np.array(intr.coeffs)  # 畸变系数
        intr_coeffs = np.array([0, 0, 0, 0, 0], dtype=np.float64)  # 畸变系数
        intr_coeffs = intr_coeffs.reshape(1, -1)

        # 检测Aruco标记
        corners, ids, rejected_img_points = aruco.detectMarkers(color_image, self.dictionary, parameters=self.parameters)
        # 估计标记的位置
        rvec, tvec, markerPoints = aruco.estimatePoseSingleMarkers(corners, 0.05, intr_matrix, intr_coeffs)

        center = None
        if ids is not None:
            print(ids)
            # corners
            # 在图像中绘制标记边界和轴
            aruco.drawDetectedMarkers(color_image, corners)
            cv2.drawFrameAxes(color_image, intr_matrix, intr_coeffs, rvec, tvec, 0.05)

            # 计算并显示Aruco标记中心
            for i, corner in zip(ids, corners):
                if calib:
                    x = (corner[0][0][0] + corner[0][3][0]) / 2
                    y = (corner[0][0][1] + corner[0][3][1]) / 2
                else:
                    x = (corner[0][0][0] + corner[0][2][0]) / 2
                    y = (corner[0][0][1] + corner[0][2][1]) / 2

                cv2.circle(color_image, (int(x), int(y)), 3, (0, 0, 255), -1)

                # 将像素坐标转换为相机坐标系下的坐标
                # dist_to_center = depth.get_distance(int(x), int(y))
                # x_cam, y_cam, z_cam = rs.rs2_deproject_pixel_to_point(intr, [x, y], dist_to_center)

                x_cam, y_cam, z_cam = convert_depth_to_xyz(int(x), int(y), img[int(y), int(x)], self.fx, self.fy, self.cx, self.cy)
                cv2.putText(color_image, "x: {:.3f}mm".format(x_cam), (int(x) + 50, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(color_image, "y: {:.3f}mm".format(y_cam), (int(x) + 50, int(y) + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(color_image, "z: {:.3f}mm".format(z_cam), (int(x) + 50, int(y) + 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                center = [x_cam, y_cam, z_cam]
                break  # 只需要一个标记的中心

        # 处理深度图像并将其与颜色图像堆叠
        # depth_img = np.asanyarray(depth.get_data())
        # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_img, alpha=0.219), cv2.COLORMAP_JET)
        # images = np.vstack((color_image, depth))
        cv2.imshow("image", color_image)
        cv2.imshow("depth", depth)
        cv2.waitKey(1)
        # cv2.waitKey(1)

        return color_image, center
   
    def run_calibration(self):
        """
        执行校准程序，通过移动机械臂到不同的位置并捕捉Aruco标记的位置来计算图像到机械臂的转换矩阵。
        """

        # 提示用户将Aruco标记放在机械臂的末端执行器上
        print("#################please put the aruco marker on the dobot end effector")
        time.sleep(30)

        # 定义校准点，格式为[x, y, z, f]，单位是毫米（mm）
        default_cali_points = [
                            [0, 170, 10, 0], [0, 160, 20, 0],
                            [0, 170, 10, 0],
                            [-10, 180, 15, 0],
                            [-10, 190, 0, 0],  [-10, 200, 10, 0],
                            [-10, 230, -5, 0], [-20, 230, 0, 0],
                            [-20, 250, 70, 0],

                            [10, 180, 20, 0], [30, 170, 10, 0],
                            [60, 160, 0, 0],
                            [40, 170, -5, 0],
                            [5, 180, 0, 0], [-15, 180, 10, 0],
                            [-25, 180, 10, 0], [-35, 180, 0, 0],
                            [-50, 180, 10, 0],

                            [120, 240, 0, 0], [125, 230, 10, 0],
                            [110, 220, 0, 0],
                            [100, 200, 5, 0],
                            [90, 180, -10, 0], [70, 160, 0, 0],
                            [80, 150, -15, 0],

                            [60, 250, -10, 0], [50, 240, 0, 0],
                            [40, 230, 5, 0],
                            [30, 220, 10, 0],
                            [10, 210, 20, 0], [0, 200, -10, 0],
                            [0, 180, 0, 0], [0, 180, 10, 0],
                            [-10, 170, 10, 0],

                            [-20, 200, 10, 0], [-30, 210, -5, 0],
                            [-30, 220, -10, 0],
                            [-30, 240, -5, 0],
                            [-40, 250, -5, 0], [-50, 250, -10, 0],
                            [-60, 250, -5, 0], [-70, 240, -10, 0],
                            [-80, 230, 0, 0],

                            [-90, 230, -15, 0],  [-100, 210, -15, 0],
                            [-110, 200, 0, 0],
                            [-120, 190, -15, 0],
                            [-120, 240, 0, 0],  [-30, 180, 10, 0],
                            [-20, 170, 20, 0], [-10, 180, 0, 0],
                            [0, 180, 10, 0],
                            ]
                            
        np_cali_points = np.array(default_cali_points)
        # 准备校准数据，将其转换为适合计算的格式
        arm_cord = np.column_stack(
            (np_cali_points[:, 0:3], np.ones(np_cali_points.shape[0]).T)).T 
        centers = np.ones(arm_cord.shape)
        
        # 定义存储转换矩阵的文件路径
        img_to_arm_file = "save_parms/image_to_arm.npy"
        arm_to_img_file = "save_parms/arm_to_image.npy"

        # 检查是否已经存在之前的校准矩阵文件，如果存在则直接加载
        if os.path.exists(img_to_arm_file) and os.path.exists(arm_to_img_file):
            image_to_arm = np.load(img_to_arm_file)
            arm_to_image = np.load(arm_to_img_file)
            print("load image to arm and arm to image transform matrix")
        else:
            # 如果不存在校准矩阵文件，则开始校准流程
            print("need to calibrate the camera and dobot")
            for index, point in enumerate(default_cali_points):
                print("#################dobot move to point {}, x: {}, y: {}, z: {}, f: {}".format(index, point[0],
                                                                                                   point[1], point[2],
                                                                                                   point[3]))
                # 将移动命令转换为Gcode格式
                # gcode_command = f"G1 X{point[0]} Y{point[1]} Z{point[2]} F{point[3]}"
                gcode_command = f"G1 X{point[0]} Y{point[1]} Z{point[2]} "
                action.send_message(gcode_command)
                time.sleep(0.5)
                # self.ser.write((gcode_command + "\r").encode("utf-8"))
                # 调整坐标，考虑Aruco标记中心与机械臂末端执行器之间的距离
                arm_cord.T[index][1] = arm_cord.T[index][1] + 35

                # 获取Aruco标记中心位置
                images, center = self.get_aruco_center()
                time.sleep(0.5)
                if center is not None:
                    # 保存中心位置
                    centers[0:3, index] = center

                else:
                    print("no aruco marker detected")
                    continue
                    # 显示图像
                # cv2.imshow("image", images)


                time.sleep(1)

        # 计算转换矩阵
        image_to_arm = np.dot(arm_cord, np.linalg.pinv(centers))
        arm_to_image = np.linalg.pinv(image_to_arm)
        print("Finished calibration!")

        print("Image to arm transform:\n", image_to_arm)
        print("Arm to Image transform:\n", arm_to_image)
        # 将转换矩阵保存到文件
        np.save(img_to_arm_file, image_to_arm)
        np.save(arm_to_img_file, arm_to_image)

        # 校验转换矩阵的正确性
        print("Sanity Test:")

        print("-------------------")
        print("Image_to_Arm")
        print("-------------------")
        for ind, pt in enumerate(centers.T):
            print("Expected:", arm_cord.T[ind][0:3])
            print("Result:", np.dot(image_to_arm, np.array(pt))[0:3])

        print("-------------------")
        print("Arm_to_Image")
        print("-------------------")
        for ind, pt in enumerate(arm_cord.T):
            print("Expected:", centers.T[ind][0:3])
            pt[3] = 1
            print("Result:", np.dot(arm_to_image, np.array(pt))[0:3])

    def run_recog(self):
        # 行识别并移动机械臂到目标位置的函数。
        # 该函数首先检查是否存在先前保存的图像到机械臂坐标转换参数，然后开启机械臂的吸取功能，并暂停3秒。
        # 接着，不断循环检测ARUCO标记中心，一旦检测到，会将当前图像保存并显示，并计算出机械臂需要移动到的目标位置。
        # 如果目标位置超出机械臂可达范围，则警告并重试。在成功到达目标位置后，执行吸取-移动-释放的动作序列。
        # 加载图像到机械臂的转换参数
        if os.path.exists("save_parms/image_to_arm.npy"):
            image_to_arm = np.load("save_parms/image_to_arm.npy")
        # self.device.suck(enable=False)  # 禁用吸取功能
        time.sleep(3)  # 暂停3秒

        while True:
            # 获取ARUCO标记中心
            images, center = self.get_aruco_center(calib=False)
            if center is not None:
                # 保存并显示当前图像
                cv2.imwrite("save.jpg", images)
                cv2.imshow("image", images)
                cv2.waitKey(1)
                
                # 计算机械臂需要移动到的目标位置
                img_pos = np.ones(4)
                img_pos[0:3] = center

                arm_pos = np.dot(image_to_arm, np.array(img_pos))
                print(arm_pos)
                
                # 检查目标位置是否超出机械臂可达范围
                # if (np.sqrt(arm_pos[0]*arm_pos[0] + arm_pos[1]*arm_pos[1]) > 300):
                #     print("Can not reach!!!!!!!!!!!!!!!")
                #     time.sleep(3)
                #     continue  # 超出范围则重试
                user_input = input("请输入一个字符：")
                if user_input == 'q':
                    print("Can not reach!!!!!!!!!!!!!!!")
                    time.sleep(3)
                    continue  # 超出范围则重试

                # 移动到目标位置
                gcode_command = f"G1 X{arm_pos[0]} Y{arm_pos[1]} Z{arm_pos[2]+20} "
                action.send_message(gcode_command)
                # 调整位置以释放物体

                gcode_command = f"G1 X{0} Y{185} Z{160} "
                action.send_message(gcode_command)
                # self.device.suck(enable=False)
                time.sleep(5)  # 等待一段时间后继续下一个目标
                print("another one")


if __name__ == "__main__":
    action = Ser("COM5")
    t1_send = threading.Thread(target=action.send_message, args=("G28",))
    # 创建一个线程，运行发送信息的函数send_message()
    t2_receive = threading.Thread(target=action.receive_message)
    # 创建一个线程，运行接收信息的函数receive_message()
    t1_send.start()  # 启动线程
    t2_receive.start()  # 启动线程
    time.sleep(1)
    # action.send_message("M114")  # 发送当前的位置信息指令

    cali = Calibration()
    if not os.path.exists("save_parms/image_to_arm.npy") or not os.path.exists("save_parms/arm_to_image.npy"):
        cali.run_calibration()
    cali.run_recog()
    # cv2.namedWindow('depth')
    # cv2.namedWindow('image')
    # 测试
    # while True:
    #     # 获取ARUCO标记中心
    #     color_image, depth, center = cali.get_aruco_center(calib=False)
    #     if center is not None:
    #         # 保存并显示当前图像
    #         # cv2.imwrite("save.jpg", images)
    #
    #         cv2.waitKey(1)
    #         img_pos = np.ones(4)
    #         img_pos[0:3] = center
    #         print("aruco center", np.array(img_pos))
    #     key = cv2.waitKey(10)
    #     if int(key) == 113:
    #         break
            # 计算机械臂需要移动到的目标位置
            # img_pos = np.ones(4)
            # img_pos[0:3] = center
            # arm_pos = np.dot(image_to_arm, np.array(img_pos))
            # print(arm_pos)
