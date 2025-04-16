import os
import cv2
import time
import threading
from queue import Queue
import config as cfg
from rknnpool import rknnPoolExecutor
import func_multithread
from detection_thread import process_detection_results, ensure_dir, process_image, setup_video_writer
from serial_thread import serial_sender
from video_writer_thread import VideoWriterThread

class VideoProcessor:
    def __init__(self, args):
        """初始化视频处理器"""
        self.args = args
        self.cap = None
        self.pool = None
        self.video_writer = None
        self.video_writer_thread = None
        self.detection_thread = None
        self.serial_thread = None
        
        # 初始化队列
        self.detection_queue = Queue()
        self.serial_queue = Queue()
        self.frame_buffer = Queue(maxsize=32)
        
        # 更新配置
        self._update_config()
        
    def _update_config(self):
        """更新配置信息"""
        self._print_args()
        self._update_save_config()
        self._update_source_config()
        
    def _print_args(self):
        """打印命令行参数"""
        print(f"命令行参数:")
        print(f"输入源: {self.args.source}")
        print(f"输入路径: {self.args.input}")
        print(f"摄像头ID: {self.args.camera_id}")
        print(f"保存视频: {self.args.save_video}")
        print(f"视频帧率: {self.args.video_fps}")
        print(f"视频编码: {self.args.video_codec}")
        
    def _update_save_config(self):
        """更新保存相关配置"""
        cfg.SaveConfig.VIDEO_SAVE_OPTIONS.update({
            'ENABLED': self.args.save_video,
            'FPS': self.args.video_fps,
            'CODEC': self.args.video_codec
        })
        cfg.SaveConfig.ENABLE_SAVE_VIDEO = self.args.save_video
        cfg.SaveConfig.VIDEO_CODEC = self.args.video_codec
        cfg.SaveConfig.VIDEO_FPS = self.args.video_fps
        
    def _update_source_config(self):
        """更新输入源配置"""
        if self.args.source == 'camera':
            cfg.SourceConfig.CURRENT_SOURCE = cfg.SourceConfig.SOURCE_TYPE['CAMERA']
            cfg.SourceConfig.CAMERA_ID = self.args.camera_id
        elif self.args.source == 'video':
            cfg.SourceConfig.CURRENT_SOURCE = cfg.SourceConfig.SOURCE_TYPE['VIDEO']
            if self.args.input:
                cfg.SourceConfig.VIDEO_PATH = self.args.input
        elif self.args.source == 'image':
            cfg.SourceConfig.CURRENT_SOURCE = cfg.SourceConfig.SOURCE_TYPE['IMAGE']
            if self.args.input:
                cfg.SourceConfig.IMAGE_PATH = self.args.input
                
    def setup(self):
        """初始化所有组件"""
        self._setup_input_source()
        self._setup_rknn_pool()
        self._setup_threads()
        self._setup_video_writer()
        
    def _setup_input_source(self):
        """设置输入源"""
        try:
            if cfg.SourceConfig.CURRENT_SOURCE == cfg.SourceConfig.SOURCE_TYPE['CAMERA']:
                # 设置设备参数
                os.system(f"v4l2-ctl --device=/dev/video-camera0 \
                          --set-fmt-video=width=640,height=480,pixelformat=NV12")
                os.system(f"v4l2-ctl --device=/dev/video-camera0 --set-parm=30")
                
                # 使用 GStreamer pipeline
                pipeline = (
                    f"v4l2src device=/dev/video-camera0 ! "
                    f"video/x-raw,format=NV12,width=640,height=480,framerate=30/1 ! "
                    f"videoconvert ! video/x-raw,format=BGR ! "
                    f"appsink"
                )
                
                # 尝试使用 GStreamer pipeline
                self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
                
                # 如果 GStreamer 失败，尝试直接打开
                if not self.cap.isOpened():
                    print("GStreamer pipeline 失败，尝试直接打开设备")
                    self.cap = cv2.VideoCapture(cfg.SourceConfig.CAMERA_ID)
                    
                    # 设置摄像头属性
                    self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('N', 'V', '1', '2'))
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    self.cap.set(cv2.CAP_PROP_FPS, 30)
                    
                if not self.cap.isOpened():
                    raise RuntimeError("无法打开摄像头")
                    
                # 打印实际的分辨率和帧率
                actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                actual_fps = int(self.cap.get(cv2.CAP_PROP_FPS))
                actual_fourcc = self.cap.get(cv2.CAP_PROP_FOURCC)
                
                print(f"摄像头初始化成功:")
                print(f"- 分辨率: {actual_width}x{actual_height}")
                print(f"- 帧率: {actual_fps}")
                print(f"- 格式: {chr(int(actual_fourcc)&0xFF)}{chr((int(actual_fourcc)>>8)&0xFF)}"
                      f"{chr((int(actual_fourcc)>>16)&0xFF)}{chr((int(actual_fourcc)>>24)&0xFF)}")
            else:
                source = (cfg.SourceConfig.CAMERA_ID 
                         if cfg.SourceConfig.CURRENT_SOURCE == cfg.SourceConfig.SOURCE_TYPE['CAMERA']
                         else cfg.SourceConfig.VIDEO_PATH)
                print(f"使用{'摄像头' if self.args.source == 'camera' else '视频文件'}输入: {source}")
                
                self.cap = cv2.VideoCapture(source)
                if not self.cap.isOpened():
                    raise RuntimeError(f"错误: 无法打开输入源 {source}")
        except Exception as e:
            print(f"设置输入源时出错: {e}")
                
    def _setup_rknn_pool(self):
        """设置RKNN处理池"""
        self.pool = rknnPoolExecutor(
            rknnModel=cfg.model_path,
            TPEs=cfg.TPEs,
            func=func_multithread.myFunc
        )
        
    def _setup_threads(self):
        """设置处理线程"""
        self.detection_thread = threading.Thread(
            target=process_detection_results,
            args=(self.detection_queue, self.serial_queue),
            daemon=True
        )
        self.serial_thread = threading.Thread(
            target=serial_sender,
            args=(self.serial_queue, cfg.serial),
            daemon=True
        )
        
        # 启动线程
        self.detection_thread.start()
        self.serial_thread.start()
        
        # 确保输出目录存在
        ensure_dir(cfg.SaveConfig.OUTPUT_DIR)
        ensure_dir(cfg.SaveConfig.IMAGE_DIR)
        ensure_dir(cfg.SaveConfig.VIDEO_DIR)
        
    def _setup_video_writer(self):
        """设置视频写入器"""
        if (cfg.SaveConfig.VIDEO_SAVE_OPTIONS['ENABLED'] and 
            not cfg.SourceConfig.CURRENT_SOURCE == cfg.SourceConfig.SOURCE_TYPE['IMAGE']):
            
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            video_path = os.path.join(
                cfg.SaveConfig.VIDEO_DIR,
                f'{cfg.SaveConfig.VIDEO_PREFIX}{timestamp}{cfg.SaveConfig.VIDEO_FORMAT}'
            )
            
            self.video_writer, target_size = setup_video_writer(self.cap, video_path)
            if self.video_writer is not None:
                self.video_writer_thread = VideoWriterThread(self.video_writer, target_size)
                
    def process(self):
        """处理视频/图像"""
        try:
            self.setup()
            if cfg.SourceConfig.CURRENT_SOURCE == cfg.SourceConfig.SOURCE_TYPE['IMAGE']:
                self._process_image()
            else:
                self._process_video()
        finally:
            self.release()
            
    def _process_image(self):
        """处理图像"""
        process_image(self.input_path, self.pool, self.detection_queue, self.serial_queue)
        
    def _process_video(self):
        """处理视频"""
        # 初始化线程池
        if self.cap.isOpened():
            for i in range(cfg.TPEs + 1):
                ret, frame = self.cap.read()
                if not ret:
                    return
                self.pool.put(frame)

        # 处理主循环
        frames = 0
        loop_time = time.time()
        
        while self.cap.isOpened():
            frames += 1
            ret, frame = self.cap.read()
            if not ret:
                break

            self.pool.put(frame)
            result, flag = self.pool.get()
            if not flag:
                break

            frame, boxes, scores, classes = result
            #cv2.imshow("检测结果", frame)
            
            if (cfg.SaveConfig.VIDEO_SAVE_OPTIONS['ENABLED'] and 
                self.video_writer_thread is not None):
                self.video_writer_thread.write(frame)

            self.detection_queue.put((boxes, scores, classes))
            
            # 显示帧率
            if frames % 30 == 0:
                fps = 30 / (time.time() - loop_time)
                print(f"\r当前帧率: {fps:.1f} FPS", end="")
                loop_time = time.time()
                
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    def release(self):
        """释放所有资源"""
        print("\n正在清理资源...")
        
        try:
            # 1. 先停止所有输入源
            if self.cap is not None:
                self.cap.release()
                print("已释放视频捕获")
            
            # 2. 停止线程池
            if self.pool is not None:
                self.pool.release()
                print("已释放线程池")
                
            # 3. 发送结束信号到队列
            print("发送结束信号到队列...")
            self.detection_queue.put((None, None, None))
            self.serial_queue.put(None)
            
            # 4. 释放视频写入线程（设置超时）
            if self.video_writer_thread is not None:
                try:
                    self.video_writer_thread.release()
                    print("已释放视频写入器")
                except Exception as e:
                    print(f"释放视频写入器时出错: {e}")
                    
            # 5. 等待其他线程结束
            if self.detection_thread and self.detection_thread.is_alive():
                self.detection_thread.join(timeout=2.0)
                if self.detection_thread.is_alive():
                    print("警告: 检测线程未能在超时时间内结束")
                else:
                    print("检测线程已结束")
                
            if self.serial_thread and self.serial_thread.is_alive():
                self.serial_thread.join(timeout=2.0)
                if self.serial_thread.is_alive():
                    print("警告: 串口线程未能在超时时间内结束")
                else:
                    print("串口线程已结束")
                
            # 6. 关闭串口
            if cfg.serial is not None:
                cfg.serial.close()
                print("已关闭串口")
                
        except Exception as e:
            print(f"清理资源时出错: {e}")
        finally:
            cv2.destroyAllWindows()
            print("资源清理完成")