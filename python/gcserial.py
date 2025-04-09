from periphery import Serial
from struct import pack
import time
# 申请串口资源/dev/ttyS3，设置串口波特率为115200，数据位为8，无校验位，停止位为1，不使用流控制
serial = Serial(
    "/dev/ttyS0",
    baudrate=9600,
    databits=8,
    parity="none",
    stopbits=1,
    xonxoff=False,
    rtscts=False,
)
# 使用申请的串口发送字节流数据 "python-periphery!\n"
value1=0
value2=0
while 1:
    value1=value1+1
    if(value1>2000):
        value1=0
    value2=value2-1
    if(value2<0):
        value2=2000

    serial.write(b'\x31' + b'\x09' + pack("<H", value1) + pack("<H", value2) + b'\x55')
    time.sleep(0.1)
     

