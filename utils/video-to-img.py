import cv2

def extract_frames(video_path, frame_interval, output_dir):
    # 读取视频文件
    video = cv2.VideoCapture(video_path)

    # 初始化帧计数器
    frame_count = 0

    while True:
        # 读取下一帧
        ret, frame = video.read()

        # 如果读取成功，保存帧到指定目录
        if ret:
            if frame_count % frame_interval == 0:
                cv2.imwrite(f'{output_dir}/frame_{frame_count}.jpg', frame)
            frame_count += 1
        else:
            break

    # 释放视频文件
    video.release()

# 使用示例
extract_frames('output1.mp4', 20, 'output_dir2')
