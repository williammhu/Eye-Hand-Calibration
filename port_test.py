# _*_ coding:utf-8 _*_
import time

import serial  # 导入serial模块，serial模块封装了对串口的访问
import threading


class Ser(object):  # 创建关于串口的类

    def __init__(self, port):
        self.port = port
        # 打开端口
        self.ser = serial.Serial(self.port, baudrate=115200, timeout=1, )
        time.sleep(1)

    def send_message(self, contunts):
        if self.ser.isOpen():  # 如果串口已经打开
            print("\nopen %s success" % self.port)  # 打印串口成功打开的信息

            if contunts:
                # com1端口向com2端口写入数据 字符串必须译码
                # self.contunts = input("输入内容:")
                #可以自定义输入Gcode命令，规划路径，例如 G1 X10 Y10 Z10 F30，其实就是三维坐标和速度
                self.ser.write((contunts+"\r").encode("utf-8"))
                time.sleep(1)
                # encode()函数是将字符串转化成相应编码方式的字节形式
                # 如str2.encode('utf-8')，表示将unicode编码的字符串str2转换成utf-8编码的字节数据。
                # 如果不转换，COM1发送到COM2的信息，COM2（调试助手）中文会识别不出来或者会出现乱码现象
        else:  # 如果没有读取到com1串口，则执行以下程序
            print("open failed")
            self.ser.close()  # 关闭端口

    def receive_message(self):
        if self.ser.isOpen():  # 如果串口已经打开
            print("\nopen COM5 sucess！")
            # print("接收信息：")

            self.receive = True  # 建立标志
            while self.receive:  # 表示一直循行读取的内容
                receives = self.ser.readlines()
                # readlines()读取多行内容

                if receives == []:
                    # 没接受有效数据时，会一直读取到空列表，等于空列表时，利用continue语句再次重新读取
                    continue

                elif receives == [b'quiet\r\n']:  # 确定接收到这个元素后，执行以下程序
                    self.receive = False
                    print("\n已关闭COM5端口")
                    self.ser.close()  # 关闭端口
                    break  # 结束循环

                elif receives != []:  # 表示当接收数据不等于空列表[]时，执行以下程序
                    # print(receives)
                    for asr in receives:  # 运用for语句将元素提取出来
                        read = asr.decode("utf-8") # 将bytes字节数据，转化为字符数据
                        print("接收信息：")
                        print(read)
            '''
            decode()的作用是将bytes数据类型转化成str数据类型的函数,
            不论什么编码方式的二进制数据，通过decode函数后，统一编成utf-8编码格式
            因为utf-8格式是python里面的标准
            '''
        else:  # 如果没有读取到com1串口，则执行以下程序
            print("\nopen failed")
            self.ser.close()  # 关闭端口

if __name__ == '__main__':
    print("\n收发信息程序已开始！")
    action = Ser("COM3")  # 这里可更改连接的串口

    t1_send = threading.Thread(target=action.send_message)

    t2_receive = threading.Thread(target=action.receive_message)

    t2_receive.start()
    action.send_message()
    print("\n收发信息程序已关闭！")