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
        """非阻塞的帧写入"""
        if not self.running:
            return
        try:
            self.queue.put_nowait(frame.copy())  # 使用非阻塞方式写入
        except Full:  # 使用导入的 Full 异常
            pass  # 队列满时直接丢弃帧
    
    def release(self):
        """释放资源"""
        self.running = False
        self.queue.put(None)  # 发送结束信号
        if self.thread.is_alive():
            self.thread.join()
        if self.writer is not None:
            self.writer.release()