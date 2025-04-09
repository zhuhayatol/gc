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
from detection_thread import process_detection_results,ensure_dir,process_image,setup_video_writer,parse_args,frame_writer_thread
from serial_thread import serial_sender
from video_writer_thread import VideoWriterThread
# 创建消息队列用于线程间通信
# detection_queue 作用是存储检测结果
# serial_queue 作用是存储串口发送的数据
detection_queue = Queue()
serial_queue = Queue()
# 添加帧缓冲队列
frame_buffer = Queue(maxsize=32)  # 限制队列大小以防内存溢出
# 串口配置
# 申请串口资源/dev/ttyS0，设置串口波特率为9600，数据位为8，无校验位，停止位为1，不使用流控制
serial=cfg.serial


def main():
    # 解析命令行参数
    args = parse_args()
    print(f"命令行参数:")
    print(f"输入源: {args.source}")
    print(f"输入路径: {args.input}")
    print(f"摄像头ID: {args.camera_id}")
    print(f"保存视频: {args.save_video}")
    print(f"视频帧率: {args.video_fps}")
    print(f"视频编码: {args.video_codec}")
    
    # 更新视频保存配置
    # 如果保存视频，则更新视频保存配置
    cfg.SaveConfig.VIDEO_SAVE_OPTIONS.update({
        'ENABLED': args.save_video,
        'FPS': args.video_fps,
        'CODEC': args.video_codec
    })
    cfg.SaveConfig.ENABLE_SAVE_VIDEO = args.save_video
    cfg.SaveConfig.VIDEO_CODEC = args.video_codec
    cfg.SaveConfig.VIDEO_FPS = args.video_fps
    
    # 更新输入源配置
    #如果输入源为摄像头，则使用摄像头ID
    #如果输入源为视频，则使用视频路径
    #如果输入源为图像，则使用图像路径
    #根据输入源类型更新配置
    if args.source == 'camera':
        cfg.SourceConfig.CURRENT_SOURCE = cfg.SourceConfig.SOURCE_TYPE['CAMERA']
        cfg.SourceConfig.CAMERA_ID = args.camera_id
    elif args.source == 'video':
        cfg.SourceConfig.CURRENT_SOURCE = cfg.SourceConfig.SOURCE_TYPE['VIDEO']
        if args.input:
            cfg.SourceConfig.VIDEO_PATH = args.input
    elif args.source == 'image':
        cfg.SourceConfig.CURRENT_SOURCE = cfg.SourceConfig.SOURCE_TYPE['IMAGE']
        if args.input:
            cfg.SourceConfig.IMAGE_PATH = args.input

    # 根据配置选择输入源
    # 如果当前输入源为图像，则不需要打开视频捕捉
    # 如果当前输入源为摄像头或视频，则需要打开视频捕捉
    # 如果当前输入源为图像，则直接使用图像路径
    # 如果当前输入源为摄像头或视频，则使用cv2.VideoCapture打开视频捕捉

    if cfg.SourceConfig.CURRENT_SOURCE == cfg.SourceConfig.SOURCE_TYPE['IMAGE']:
        cap = None
        input_path = cfg.SourceConfig.IMAGE_PATH
        print(f"使用图像输入: {input_path}")
    else:
        if cfg.SourceConfig.CURRENT_SOURCE == cfg.SourceConfig.SOURCE_TYPE['CAMERA']:
            source = cfg.SourceConfig.CAMERA_ID
            print(f"使用摄像头输入, ID: {source}")
        else:
            source = cfg.SourceConfig.VIDEO_PATH
            print(f"使用视频文件输入: {source}")
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"错误: 无法打开输入源 {source}")
            return

    # 初始化rknn池
    #myFunc是一个函数，用于处理视频帧，该函数位于func_multithread.py文件中
    pool = rknnPoolExecutor(
        rknnModel=cfg.model_path,
        TPEs=cfg.TPEs,
        func=func_multithread.myFunc)
    
    # 启动处理线程和串口线程
    detection_thread = threading.Thread(
        target=process_detection_results,
        args=(detection_queue, serial_queue),
        daemon=True
    )
    serial_thread = threading.Thread(
        target=serial_sender,
        args=(serial_queue, serial),
        daemon=True
    )
    detection_thread.start()
    serial_thread.start()
    
    # 确保输出目录存在
    ensure_dir(cfg.SaveConfig.OUTPUT_DIR)
    ensure_dir(cfg.SaveConfig.IMAGE_DIR)
    ensure_dir(cfg.SaveConfig.VIDEO_DIR)

    # 获取当前时间戳作为文件名
    timestamp = time.strftime("%Y%m%d_%H%M%S")

# 只在需要时创建视频写入器
    video_writer = None
    video_writer_thread = None
    target_size = None
    # 如果启用视频保存功能，并且当前输入源不是图像
    if (cfg.SaveConfig.VIDEO_SAVE_OPTIONS['ENABLED'] and 
        not cfg.SourceConfig.CURRENT_SOURCE == cfg.SourceConfig.SOURCE_TYPE['IMAGE']):
        video_path = os.path.join(
            cfg.SaveConfig.VIDEO_DIR,
            f'{cfg.SaveConfig.VIDEO_PREFIX}{timestamp}{cfg.SaveConfig.VIDEO_FORMAT}'
        )
        video_writer, target_size = setup_video_writer(cap, video_path)
        if video_writer is not None:
            video_writer_thread = VideoWriterThread(video_writer, target_size)

    # 在此之前的代码是初始化模型和参数
    # 主线程循环作用是读取视频帧并将其放入线程池中进行处理
    try:
        if cfg.SourceConfig.CURRENT_SOURCE == cfg.SourceConfig.SOURCE_TYPE['IMAGE']:
            # 处理单张图片
            process_image(input_path, pool, detection_queue, serial_queue)

            

        else:
            # 初始化异步所需要的帧
            # 要确保线程池中始终有足够的任务，避免线程空闲
            if (cap.isOpened()):
                for i in range(cfg.TPEs + 1):
                    ret, frame = cap.read()
                    if not ret:
                        cap.release()
                        del pool
                        exit(-1)
                    pool.put(frame)

            frames, loopTime, initTime = 0, time.time(), time.time()
            # 主循环
            while (cap.isOpened()):
                frames += 1
                ret, frame = cap.read()
                if not ret:
                    break
                
                pool.put(frame)
                result, flag = pool.get()
                if flag == False:
                    break

                frame, boxes, scores, classes = result
                
                # 简化的视频写入逻辑
                if (cfg.SaveConfig.VIDEO_SAVE_OPTIONS['ENABLED'] and 
                    video_writer_thread is not None):
                    video_writer_thread.write(frame)  # 非阻塞写入

                detection_queue.put((boxes, scores, classes))
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                if frames % 30 == 0:
                    print("30帧平均帧率:\t", 30 / (time.time() - loopTime), "帧")
                    loopTime = time.time()

    finally:
        print("正在清理资源...")
        
        # 1. 首先停止输入
        if cap is not None:
            cap.release()
            print("已释放视频捕获")
            
        # 2. 停止线程池
        if pool is not None:
            pool.release()
            print("已释放线程池")
            
        # 3. 停止视频写入
        if video_writer_thread is not None:
            video_writer_thread.release()
            print("已释放视频写入器")
            
        # 4. 发送结束信号给其他线程
        detection_queue.put((None, None, None))
        serial_queue.put(None)
        
        # 5. 等待其他线程结束
        if detection_thread and detection_thread.is_alive():
            detection_thread.join(timeout=2.0)
            print("检测线程已结束")
            
        if serial_thread and serial_thread.is_alive():
            serial_thread.join(timeout=2.0)
            print("串口线程已结束")
            
        # 6. 释放其他资源
        if serial is not None:
            serial.close()
            print("已关闭串口")
            
        cv2.destroyAllWindows()
        print("资源清理完成")
        
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"程序发生错误: {e}")
    finally:
        print("程序退出")