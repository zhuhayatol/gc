import cv2
from queue import Queue, Empty, Full  # 正确导入 Empty 和 Full 异常
import threading
import time

class VideoWriterThread:
    def __init__(self, writer, target_size=None):
        self.writer = writer
        self.target_size = target_size
        self.queue = Queue(maxsize=64)  # 设置合适的队列大小
        self.running = True
        self.thread = threading.Thread(target=self._write_frames, daemon=True)
        self.thread.start()
    
    def _write_frames(self):
        """简单的写入循环"""
        while self.running:
            try:
                frame = self.queue.get(timeout=0.1)  # 设置超时避免CPU空转
                if frame is None:  # 结束信号
                    break
                if self.target_size is not None:
                    frame = cv2.resize(frame, self.target_size)
                self.writer.write(frame)
            except Empty:  # 使用导入的 Empty 异常
                continue
            except Exception as e:
                print(f"视频写入错误: {str(e)}")
    
    def write(self, frame):
        """阻塞方式写入帧到队列"""
        if not self.running:
            return False
        try:
            # 使用阻塞方式写入，timeout设置等待时间
            self.queue.put(frame.copy(), timeout=1.0)  
            return True
        except Full:
            print("警告: 视频写入队列已满，等待写入...")
            try:
                # 再次尝试写入
                self.queue.put(frame.copy(), block=True)
                return True
            except:
                print("错误: 写入失败")
                return False
    
    def release(self):
        """安全释放资源"""
        if not self.running:
            return
            
        print("正在关闭视频写入线程...")
        self.running = False
        
        try:
            # 1. 清空队列
            while not self.queue.empty():
                try:
                    _ = self.queue.get_nowait()
                except Empty:
                    break
                    
            # 2. 发送结束信号
            try:
                self.queue.put(None, timeout=1.0)
            except Full:
                print("警告: 无法发送结束信号")
                
            # 3. 等待线程结束（设置超时）
            if self.thread.is_alive():
                print("等待视频写入线程结束...")
                self.thread.join(timeout=3.0)
                
                if self.thread.is_alive():
                    print("警告: 视频写入线程未能在超时时间内结束")
                    
            # 4. 释放写入器
            if self.writer is not None:
                self.writer.release()
                print("视频写入器已释放")
                
        except Exception as e:
            print(f"释放视频写入器时出错: {e}")
        finally:
            print("视频写入线程清理完成")