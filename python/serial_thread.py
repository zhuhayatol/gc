import time
from struct import pack

def serial_sender(serial_queue, serial):
    """串口发送线程"""
    while True:
        if serial_queue.empty():
            time.sleep(0.001)
            continue
            
        message = serial_queue.get()
        if message is None:  # 退出信号
            break
            
        try:
            # 发送串口数据
            serial.write(
                b'\x31'+
                message['command']  +
                pack("<H", message['x']) +
                pack("<H", message['y']) +
                b'\x55'
            )

            print(f"Serial sent: {message['command']} {message['x']} {message['y']}")
            
            time.sleep(0.01)  # 控制发送频率
        except Exception as e:
            print(f"Serial error: {e}")