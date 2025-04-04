#导入必须的功能包
from maix import camera, display, image, nn, app,time,uart,pwm,pinmap,gpio
from struct import pack
import threading
from types import SimpleNamespace
#导入yolo11训练转换好的模型
# detector = nn.YOLO11(model="/root/models/yolo11_2600_int8.mud", dual_buff = True)
detector = nn.YOLO11(model="/root/models/yolo11_5500_736_512/yolo11_5500.mud", dual_buff = True)
#寻找串口设备
device = "/dev/ttyS0"
#设定摄像头的输入图像尺寸
input_width=960
input_height=672
cam = camera.Camera(input_width, input_height, detector.input_format())
cam.skip_frames(30) #跳过开头的30帧
disp = display.Display()
#设定波特率
serial = uart.UART(device, 9600)
#设定球颜色的顺序
color_order = ["blue ball","red ball","yellow ball","black ball","blue square","red square","blue start","red start"]
ball_come_flag=0
#创建一个命名空间管理所有变量
robot_state = SimpleNamespace()
robot_state.max_idle_time=5 #10秒的空闲时间（单位：毫秒）,没找到就切换下一个球目标
robot_state.ball_unimport = 0 #不需要关注的球(不需要关注蓝球)
robot_state.ball_need = not robot_state.ball_unimport
robot_state.target_ball = color_order[robot_state.ball_need]  #需要关注的球
robot_state.target_ball_index = 2 #需要关注的球的序号，初始关注黄球
robot_state.last_detection_time = 0 #上次记录到球时的时间
robot_state.tag_findred = False #标志位：球是否位于视野下方
robot_state.tag_findsquare = False #标志位：执行寻找球或者寻找安全区的标志位，False为寻找球
robot_state.ball_place = False #决定相机位置的标志位
robot_state.zhuazi_place = False #决定爪子开合的标志位
robot_state.center_x = 210 #球或者安全区中心的横坐标
robot_state.center_y = 134 #球或者安全区中心的纵坐标
robot_state.message = b'\x02' #上位机需要给下位机发送的信息
robot_state.FIRST_task = False #完成了寻找第一个基础球的任务
robot_state.img=cam.read() #读取到的图像
robot_state.duoji_tag=False #舵机互斥锁，舵机执行的时候yolo主任务不会执行
robot_state.done=False #完成了一次寻球，进入判断球是否在框内的函数的标志位
robot_state.start=time.time()
robot_state.start_tag=False
robot_state.TL=0
x1, y1 = 350, (5*input_height)//7  # 区域左上角
x2, y2 = input_width-350, input_height  # 区域右下角

# #照明
pinmap.set_pin_function("B3", "GPIOB3")
led = gpio.GPIO("GPIOB3", gpio.Mode.OUT)
led.value(1)


#PWM的参数
SERVO_PERIOD = 50     # 50Hz 20ms
SERVO_MIN_DUTY = 2.5  # 2.5% -> 0.5ms
SERVO_MAX_DUTY = 12.5  # 12.5% -> 2.5ms
pwm_id1 = 6
# !! set pinmap to use PWM6
pinmap.set_pin_function("A18", "PWM6")
pwm_id2 = 7
# !! set pinmap to use PWM7
pinmap.set_pin_function("A19", "PWM7")

#舵机转动定义
'''
舵机分为上面的相机转动舵机与下面的爪子抓取舵机,分别记为out1与out2
out1初始位置为60,俯视时为30,当相机平视且小球在相机视野下半部分时则转为俯视
当俯视时小球仍然在相机视野下半部分时转为平视并且去寻找安全区
out2初始位置为爪子张开角度为[]-1,爪子抓住时为[]-2,初始时为爪子张开
'''
def angle_to_duty(percent):
    return (SERVO_MAX_DUTY - SERVO_MIN_DUTY) * percent / 100 + SERVO_MIN_DUTY
out1 = pwm.PWM(pwm_id1, freq=SERVO_PERIOD, duty=angle_to_duty(90), enable=True)
out2 = pwm.PWM(pwm_id2, freq=SERVO_PERIOD, duty=angle_to_duty(56), enable=True)

def duoji_rotation_pos(out,percent_low,percent_high):
    for i in range(percent_low,percent_high):
        out.duty(angle_to_duty(i))
        # time.sleep_ms(1)
def duoji_rotation_neg(out,percent_low,percent_high):
    for i in range(percent_low,percent_high):
        out.duty(angle_to_duty(100-i))
        # time.sleep_ms(1)

'''
考虑到上位机需要给下位机发信，现规定发信协议：
帧头:0x7B;运动模式信息:msg;坐标信息:center_x;坐标信息:center_y帧尾:0x7Ds
msg种类:
0x01:寻找目标模式（位置环）
0x02:旋转模式
0x03:倒退模式
0x04:停车模式
0x05:向90°旋转
0x06:慢走
'''

#寻找需要的球并且完成任务（注：这是第一次任务，需要完成第一次寻找己方球的任务才能去寻找高价值的黄、黑球）
def find_ball_need():
    """
    只关注需要找到的球，直到找到那个球才停止。
    :return: 是否找到那个球
    """
    robot_state.target_ball = color_order[robot_state.ball_need] #确定目标球
    center_x1=0
    center_y1=0
    finished = False  # 用于标记是否检测到红球
    done=False
    # 读取摄像头图像
    robot_state.img = cam.read()
    objs = detector.detect(robot_state.img, conf_th=0.3, iou_th=0.45)
    message=b'\x02'#命令为旋转模式
    # 执行物体检测
    for obj in objs:
        # 检测到目标小球
        if detector.labels[obj.class_id] == robot_state.target_ball:
           
            if  pow(obj.y + obj.h // 2,2)+pow(obj.x + obj.w // 2,2)>pow(center_x1,2)+pow(center_y1,2):
                center_x1 = obj.x + obj.w // 2
                center_y1 = obj.y + obj.h // 2 #确定球的中心位置
            if robot_state.ball_place == True:
                message=b'\x06'
            else:
                message=b'\x01'
    if time.time()-robot_state.last_detection_time>=5:
            if robot_state.ball_place==True:
                robot_state.ball_place=False
                robot_state.tag_findred = False
                robot_state.last_detection_time=time.time()
            elif robot_state.ball_place==False:
                robot_state.ball_place=True
                robot_state.last_detection_time=time.time()                
    if robot_state.tag_findred == True: 
        if x1 <= center_x1 <= x2 and center_y1 >= y1+60 and time.time()-robot_state.last_detection_time>=1:
            #控制下方爪子抓取函数
            robot_state.tag_findsquare=True
            finished = True 
            robot_state.ball_place=False
            robot_state.zhuazi_place=True
            message=b'\x04'
            print("2")
            done=True
    #靠的很近
    if x1 <= center_x1 <= x2 and (5*input_height)//7 <= center_y1 <= y2 and not robot_state.tag_findred: 
        robot_state.ball_place=True
        robot_state.tag_findred=True
        print("3")
        robot_state.last_detection_time=time.time()
    return message, finished, center_x1, center_y1,done

last_detection_time=0
#寻找安全区任务
def find_safe_place():
    """
    控制小车将抓到的蓝球放入蓝方的安全区。
    :param center_x: 小车当前抓取的蓝球中心x坐标
    :param center_y: 小车当前抓取的蓝球中心y坐标
    :return: 是否完成任务
    """
    global  last_detection_time
    robot_state.img = cam.read()
    message=b'\x02'
    center_sq_x=0
    center_sq_y=0
    #安全区的坐标
    squarey1=0
    squarey2=0
    target_square = color_order[robot_state.ball_need+4]
    objs = detector.detect(robot_state.img, conf_th=0.5, iou_th=0.45)
    detect=False
    for obj in objs:
        if detector.labels[obj.class_id] == target_square:
            # robot_state.img.draw_rect(obj.x, obj.y, obj.w, obj.h, color=image.COLOR_RED)
            # msg = f'{target_square}: {obj.score:.2f}'
            # robot_state.img.draw_string(obj.x, obj.y, msg, color=image.COLOR_RED)#画框并且标注信息
            squarey1 = obj.y
            squarey2 = obj.y + obj.h
            center_sq_x = obj.x + obj.w//2
            center_sq_y = obj.y + obj.h//2
            message=b'\x01'
            robot_state.last_detection_time=time.time()
            detect=True
        if detector.labels[obj.class_id] == color_order[robot_state.ball_unimport+4]:
            message=b'\x05'
            return message,center_sq_x,center_sq_y
    # 判断球是否已经到达安全区内
    if squarey2>=6*input_height/7 and not robot_state.ball_place:
        robot_state.ball_place=True
        last_detection_time=time.time()
        print("get down")
    
    if robot_state.ball_place and time.time()-last_detection_time>=1.5 :
        if not detect and time.time()-robot_state.last_detection_time>=3:
             robot_state.ball_place=False
        if squarey2>=7*input_height/8:
            robot_state.ball_place=False
            robot_state.tag_findsquare=False
            robot_state.tag_findred=False
            robot_state.zhuazi_place=False
            message = b'\x03'  # 停车的命令
            robot_state.target_ball_index=2
    if message==b'\x01':
        if robot_state.ball_place==True:
            message=b'\x06'
        else:
            message=b'\x01'
    return message,center_sq_x,center_sq_y
start_tag=False
def find_target_ball():
    """
    检测目标小球，并判断是否超时切换到下一个目标小球。
    :param target_ball_index: 当前关注的小球索引
    :param last_detection_time: 上次检测目标小球的时间
    :param max_idle_time: 超过该时间则切换目标小球
    :return: 新的目标小球索引，更新后的上次检测时间，是否检测到目标小球
    """
    global  last_detection_time
    center_x=960
    center_y=0
    detect=False
    done=False
    # 读取摄像头图像
    robot_state.img=0
    robot_state.img = cam.read()
    message=b'\x02'
    # 执行物体检测
    objs = detector.detect(robot_state.img, conf_th=0.3, iou_th=0.45)
    #安全区的坐标
    squarex1,squarey1=0,0 #安全区的顶点左边
    sq_weight,sq_height=0,0 #安全区的宽
    tag_sq=False
    # for obj in objs:
    #     if ((detector.labels[obj.class_id] == color_order[6] ) or (detector.labels[obj.class_id] == color_order[7] )) and len(objs)>=2:
    #             if obj.y+obj.h>=(3*input_height)//4:
    #                 robot_state.start_tag =True
    #                 robot_state.TL=time.time()
    #                 print("start")

    for obj in objs:
        if ((detector.labels[obj.class_id] == color_order[4] ) or (detector.labels[obj.class_id] == color_order[5])):
        # if ((detector.labels[obj.class_id] == color_order[4] ) or (detector.labels[obj.class_id] == color_order[5])) and (time.time()-robot_state.TL>=6 or not robot_state.start_tag):
            squarex1 = obj.x
            squarey1 = obj.y
            sq_weight=obj.w
            sq_height=obj.h-50
            tag_sq=True
            # robot_state.start_tag=False
            print("non")
    if robot_state.ball_place==True:
        for obj in objs:
            if detector.labels[obj.class_id] == color_order[robot_state.ball_need] and \
            (tag_sq and not (squarey1<= obj.y + (obj.h // 2)<=squarey1+sq_height and squarex1<= obj.x + (obj.w // 2)<=squarex1+sq_weight) or not tag_sq):
                if  pow(obj.y + obj.h // 2,2)+pow(obj.x + obj.w // 2-960,2)>pow(center_x-960,2)+pow(center_y,2):
                        center_x = obj.x + obj.w // 2
                        center_y = obj.y + obj.h // 2 #确定球的中心位置
                        robot_state.last_detection_time=time.time()
                        message=b'\x01'
        for obj in objs:
            # 检测到目标小球
            if detector.labels[obj.class_id]==color_order[2] or detector.labels[obj.class_id]==color_order[3]:
                # 绘制框和标签
                if (tag_sq  and (not (squarey1<= obj.y + (obj.h // 2)<=squarey1+sq_height and squarex1<= obj.x + (obj.w // 2)<=squarex1+sq_weight))) or (not tag_sq):
                    detect=True 
                    if detector.labels[obj.class_id]==color_order[2]:
                        center_x = obj.x + obj.w // 2
                        center_y = obj.y + obj.h // 2 #确定球的中心位置
                        robot_state.last_detection_time=time.time()
                        message=b'\x01'
                        break
                    if detector.labels[obj.class_id]==color_order[3]:
                        if pow(obj.y + obj.h // 2,2)+pow(obj.x + obj.w // 2-960,2)>pow(center_x-960,2)+pow(center_y,2):
                            center_x = obj.x + obj.w // 2
                            center_y = obj.y + obj.h // 2 #确定球的中心位置
                            robot_state.last_detection_time=time.time()
                            message=b'\x01'
    else:
        for obj in objs:
            if detector.labels[obj.class_id]==color_order[2] or detector.labels[obj.class_id]==color_order[3] or color_order[robot_state.ball_need]:
                if (tag_sq  and (not (squarey1<= obj.y + (obj.h // 2)<=squarey1+sq_height and squarex1<= obj.x + (obj.w // 2)<=squarex1+sq_weight))) or (not tag_sq):
                    if  pow(obj.y + obj.h // 2,2)+pow(obj.x + obj.w // 2-960,2)>pow(center_x-960,2)+pow(center_y,2):
                        center_x = obj.x + obj.w // 2
                        center_y = obj.y + obj.h // 2 #确定球的中心位置
                        robot_state.last_detection_time=time.time()
                        message=b'\x01'
    if time.time()-robot_state.last_detection_time>=5:
            if robot_state.ball_place==True:
                robot_state.ball_place=False
                robot_state.tag_findred = False
                robot_state.last_detection_time=time.time()
            elif robot_state.ball_place==False:
                robot_state.ball_place=True
                robot_state.last_detection_time=time.time()
    if (x1 <= center_x <= x2 and (5*input_height)//7 <= center_y <= y2 and not robot_state.tag_findred):
        robot_state.ball_place=True
        robot_state.tag_findred=True
        last_detection_time=time.time()
    if robot_state.tag_findred == True:
        if x1 <= center_x <= x2 and y1+60 <= center_y <= y2 and time.time()-last_detection_time>=1:
            #控制下方爪子抓取函数
            message=b'\x04'
            robot_state.ball_place=False
            robot_state.tag_findsquare=True
            robot_state.zhuazi_place=True
            done=True
    if message==b'\x01':
        if robot_state.ball_place==True:
            message=b'\x06'
        else:
            message=b'\x01'
    return  message, center_x, center_y,done


def ball_come():
    """
    确认球在不在框中的函数
    """
    robot_state.message=b'\x04'
    center_y=0
    tag_oth=False
    # 执行物体检测
    robot_state.ball_place=True
    time.sleep_ms(100)
    robot_state.message=b'\x09'
    time.sleep_ms(300)
    robot_state.message=b'\x08'
    time.sleep_ms(300)
    time_now=time.time()
    while time.time()-time_now<=0.3:
        robot_state.img = cam.read()
        objs = detector.detect(robot_state.img, conf_th=0.5, iou_th=0.45)
        robot_state.message=b'\x04'
        # if not robot_state.FIRST_task:
        #     for obj in objs:
        #         if  detector.labels[obj.class_id]==color_order[2] or detector.labels[obj.class_id]==color_order[3]:
        #             tag_oth=True
        #             break
        for obj in objs:
            if not robot_state.FIRST_task:
                if  detector.labels[obj.class_id]==color_order[robot_state.ball_need]:
                    center_x = obj.x + obj.w // 2
                    center_y = obj.y + obj.h // 2
                    if center_y>=430 and center_x>=275 and center_x<=input_width-275:
                        robot_state.done=False
                        return 1
            elif robot_state.FIRST_task: 
                if  detector.labels[obj.class_id]==color_order[2] or detector.labels[obj.class_id]==color_order[3] or detector.labels[obj.class_id]==color_order[robot_state.ball_need]:
                    center_x = obj.x + obj.w // 2
                    center_y = obj.y + obj.h // 2
                    if center_y>=430 and center_x>=275 and center_x<=input_width-275:
                        robot_state.done=False
                        return 1
                
    print("no")
    return 0

def main_yolo11():
    global tag
    tag=0
    flag_tosqure=0
    time.sleep_ms(3600)
    while not app.need_exit():
        if not robot_state.duoji_tag:
            if not robot_state.tag_findsquare:
                if not robot_state.FIRST_task:
                    # 先执行寻找红球的逻辑
                    robot_state.message, detected , robot_state.center_x ,robot_state.center_y,robot_state.done= find_ball_need()
                else:
                    robot_state.message, robot_state.center_x, robot_state.center_y,robot_state.done= find_target_ball()
            else:
                if robot_state.done==True:
                    ball_come_flag=ball_come()
                    if ball_come_flag==1:
                        flag_tosqure=1
                        #robot_state.message,robot_state.center_x,robot_state.center_y= find_safe_place()
                        robot_state.FIRST_task = True
                    else:
                        flag_tosqure=0
                        robot_state.ball_place=False
                        robot_state.tag_findsquare=False
                        robot_state.tag_findred=False
                        robot_state.zhuazi_place=False
                if flag_tosqure==1:
                    robot_state.message,robot_state.center_x,robot_state.center_y= find_safe_place()
                
            if robot_state.message==b'\x03': #当命令为倒退时，延时大约4s保证车子能退出去
                tag=time.time()
                while time.time()-tag<=1:
                    robot_state.img=cam.read()
                    detector.detect(robot_state.img, conf_th=0.5, iou_th=0.45)
                robot_state.last_detection_time=time.time()
            if robot_state.message==b'\x04': #当命令为停止时，停车1s左右
                tag=time.time()
                while time.time()-tag<=1:
                    robot_state.img=cam.read()
                    detector.detect(robot_state.img, conf_th=0.5, iou_th=0.45)
        # fps=time.fps()
        # print(fps)
    
    
#串口发送线程
def uart_send_thread():
    while not app.need_exit():
        t_n=time.time()
        if t_n-robot_state.start<=3:
            serial.write(b'\x31' + b'\x07' + pack("<H", robot_state.center_x) + pack("<H", robot_state.center_y) + b'\x55')
        else:
            if robot_state.message==b'\x05':
                serial.write(b'\x31' + robot_state.message + pack("<H", robot_state.center_x) + pack("<H", robot_state.center_y) + b'\x55')
                time.sleep_ms(2500)
            else:
                serial.write(b'\x31' + robot_state.message + pack("<H", robot_state.center_x) + pack("<H", robot_state.center_y) + b'\x55')
        time.sleep_ms(10)

# 相机舵机控制线程
def servo_control_thread():
    while not app.need_exit():
        robot_state.duoji_tag=False
        if robot_state.ball_place:
            robot_state.duoji_tag=True
            out2.duty(angle_to_duty(29))
        else :
            robot_state.duoji_tag=True
            out2.duty(angle_to_duty(50))
        robot_state.duoji_tag=False
        time.sleep_ms(20)
#执行机构控制线程
compete_tag=False
def guazi_thread():
    global compete_tag
    robot_state.duoji_tag=False
    while not app.need_exit():
        if robot_state.zhuazi_place and not compete_tag:
            robot_state.duoji_tag=True
            out1.duty(angle_to_duty(20))
            compete_tag=True
        elif not robot_state.zhuazi_place and compete_tag:
            robot_state.duoji_tag=True
            out1.duty(angle_to_duty(90))
            if robot_state.done==False:
                robot_state.message=b'\x03'
            compete_tag=False
        time.sleep_ms(20)
        robot_state.duoji_tag=False

        


threading.Thread(target=main_yolo11,daemon=True).start()
threading.Thread(target=uart_send_thread, daemon=True).start()
threading.Thread(target=servo_control_thread, daemon=True).start()
threading.Thread(target=guazi_thread, daemon=True).start()
# threading.Thread(target=LED, daemon=True).start()
# 主循环
while not app.need_exit():
    time.sleep(10)