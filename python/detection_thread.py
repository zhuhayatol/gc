import time
from queue import Queue
from struct import pack
import threading
import cv2
import os
import config as cfg
import argparse
def process_detection_results(detection_queue, serial_queue):
    """检测结果处理线程"""
    while True:
        if detection_queue.empty():
            time.sleep(0.001)
            continue
            
        boxes, scores, classes = detection_queue.get()
        if boxes is None:  # 退出信号
            break
            
        # 处理检测结果的复杂逻辑
        for box, score, cl in zip(boxes, scores, classes):
            x1, y1, x2, y2 = [int(_b) for _b in box]
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            
            # 示例：根据不同类别和位置进行判断
            
            # 将串口消息放入队列
            serial_queue.put({
                'command': b'\x31',
                'x': center_x,
                'y': center_y,
            })


def process_image(image_path, pool, detection_queue, serial_queue):
    """处理单张图片"""
    # 读取图像
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Error: Cannot read image from {image_path}")
        return
    
    # 送入线程池处理
    pool.put(frame)
    result, flag = pool.get()
    
    if not flag:
        return
    
    # 解包结果
    frame, boxes, scores, classes = result
    
    if boxes is not None:
        print("\n检测结果:")
        print("-" * 50)
        for i, (box, score, cl) in enumerate(zip(boxes, scores, classes)):
            x1, y1, x2, y2 = [int(_b) for _b in box]
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            print(f"目标 {i+1}:")
            print(f"  类别: {cfg.CLASSES[cl]}")
            print(f"      置信度: {score:.3f}")
            print(f"  边界框: ({x1}, {y1}, {x2}, {y2})")
            print(f"  中心点: ({center_x}, {center_y})")
            print("-" * 50)
    else:
        print("未检测到任何目标")
    # 显示结果
    #cv2.imshow('test', frame)
    #cv2.waitKey(0)  # 等待按键

    # 保存处理后的图像
    if cfg.SaveConfig.ENABLE_SAVE_IMAGE:
        # 生成保存路径
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(
            cfg.SaveConfig.IMAGE_DIR,
            f'{cfg.SaveConfig.IMAGE_PREFIX}{timestamp}.jpg'
        )
        # 保存图像
        cv2.imwrite(save_path, frame)
        print(f"Saved processed image to {save_path}")

    # 将检测结果放入队列
    detection_queue.put((boxes, scores, classes))

    
def ensure_dir(dir_path):
    """确保目录存在，如果不存在则创建"""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def setup_video_writer(cap, output_path):
    """设置视频写入器"""
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cfg.SaveConfig.VIDEO_FPS
    
    # 打印参数用于调试
    print(f"视频写入器参数:")
    print(f"原始分辨率: {width}x{height}")
    
    # 等比例缩放
    max_dimension = 640  # 设置最大边长
    if width > max_dimension or height > max_dimension:
        # 计算缩放比例
        scale = min(max_dimension / width, max_dimension / height)
        # 保持宽高比进行缩放
        new_width = int(width * scale)
        new_height = int(height * scale)
        # 确保宽高为偶数（某些编码器要求）
        new_width = new_width - (new_width % 2)
        new_height = new_height - (new_height % 2)
        width, height = new_width, new_height
        print(f"调整后的分辨率: {width}x{height}")
    print(f"帧率: {fps}")
    print(f"编码器: {cfg.SaveConfig.VIDEO_CODEC}")
    print(f"输出路径: {output_path}")
    

    
    try:
        fourcc = cv2.VideoWriter_fourcc(*cfg.SaveConfig.VIDEO_CODEC)
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if not writer.isOpened():
            print("错误: 无法初始化视频写入器")
            return None
            
        print("视频写入器初始化成功")
        return writer, (width, height)  # 返回写入器和目标尺寸
        
    except Exception as e:
        print(f"创建视频写入器时发生错误: {str(e)}")
        return None, None

def parse_args():
    parser = argparse.ArgumentParser(description='视频处理程序')
    # 输入源选择
    parser.add_argument('--source', 
                       type=str,
                       choices=['camera', 'video', 'image'],
                       default='camera',
                       help='选择输入源类型(camera/video/image)')
    parser.add_argument('--input',
                       type=str,
                       default='../model/15.mp4',
                       help='输入文件路径(视频或图像)')
    parser.add_argument('--camera-id',
                       type=int,
                       default=11,
                       help='摄像头ID')
    
    # 视频保存选项
    parser.add_argument('--save-video', 
                       action='store_true',
                       default=False,
                       help='启用视频保存功能')
    parser.add_argument('--video-fps',
                       type=int,
                       default=30,
                       help='保存视频的帧率')
    parser.add_argument('--video-codec',
                       type=str,
                       default='MJPG',
                       help='视频编码器(MJPG, XVID等)')
    return parser.parse_args()

# 创建帧写入线程函数
def frame_writer_thread(frame_buffer, video_writer_thread):
    """专门处理视频写入的线程"""
    while True:
        try:
            frame = frame_buffer.get()
            if frame is None:  # 结束信号
                break
                
            video_writer_thread.write(frame)
            
        except Exception as e:
            print(f"视频写入错误: {str(e)}")
            continue
