import os
import sys
import cv2
import time
import numpy as np
from rknnlite.api import RKNNLite
from rknnpool import rknnPoolExecutor
import argparse
from periphery import Serial
import func_multithread
import config as cfg
from queue import Queue
import threading
# 导入线程模块
from detection_thread import process_detection_results, ensure_dir, process_image, setup_video_writer, parse_args, frame_writer_thread
from serial_thread import serial_sender
from video_writer_thread import VideoWriterThread
from video_processor import VideoProcessor

# 创建消息队列用于线程间通信
# detection_queue 作用是存储检测结果
# serial_queue 作用是存储串口发送的数据
detection_queue = Queue()
serial_queue = Queue()
# 添加帧缓冲队列
frame_buffer = Queue(maxsize=32)  # 限制队列大小以防内存溢出
# 串口配置
# 申请串口资源/dev/ttyS0，设置串口波特率为9600，数据位为8，无校验位，停止位为1，不使用流控制
serial = cfg.serial

def main():
    args = parse_args()
    processor = VideoProcessor(args)
    processor.process()



if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"程序发生错误: {e}")
    finally:
        print("程序退出")