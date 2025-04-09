from periphery import Serial
# 定义目标检测的参数
OBJ_THRESH = 0.25  # 目标检测的置信度阈值
NMS_THRESH = 0.45  # 非极大值抑制（NMS）的阈值
IMG_SIZE = (640, 640)  # 输入图像的尺寸（宽，高）

# 感兴趣范围坐标
CENTER_X = 320
CENTER_Y = 320
WIDTH = 640
HEIGHT = 640

X1 = CENTER_X - WIDTH / 2
X2 = CENTER_X + WIDTH / 2
Y1 = CENTER_Y - HEIGHT / 2
Y2 = CENTER_Y + HEIGHT / 2


# 选择是否输出预测框坐标
OUTPUT_PREDICTION_BOX_COORDINATES = False


# 检测类别顺序列表
CLASSES = (
 "red ball","black ball","blue ball","yellow ball","red square","blue square"
)

# COCO 数据集的类别 ID 列表（与类别顺序对应）
coco_id_list = [
    0, 1, 2, 3, 4, 5 ]


# 串口配置
# 申请串口资源/dev/ttyS0，设置串口波特率为9600，数据位为8，无校验位，停止位为1，不使用流控制
serial = Serial(
    "/dev/ttyS0",
    baudrate=9600,
    databits=8,
    parity="none",
    stopbits=1,
    xonxoff=False,
    rtscts=False,
)

#model_path 是模型的路径
model_path='../model/best1.rknn'
    
#camera_id
camera_id=0

#视频文件路径
video_path='../model/15.mp4'

#图片文件路径
image_path='../model/hay.jpg'

# 线程数, 增大可提高帧率
TPEs = 3\


"""
配置文件: 视频和图像相关配置
"""
class SourceConfig:
    # 输入源类型
    SOURCE_TYPE = {
        'CAMERA': 0,
        'VIDEO': 1,
        'IMAGE': 2
    }
    
    # 路径配置
    VIDEO_PATH = '../model/15.mp4'
    IMAGE_PATH = '../model/hay.jpg'
    CAMERA_ID = 0
    
    # 当前使用的输入源类型
    CURRENT_SOURCE = SOURCE_TYPE['IMAGE']  # 设置为图像模式


class SaveConfig:
    # 是否启用保存功能
    ENABLE_SAVE_IMAGE = True
    ENABLE_SAVE_VIDEO = False  # 默认关闭视频保存
    
    # 保存路径配置
    OUTPUT_DIR = '../output'
    IMAGE_DIR = f'{OUTPUT_DIR}/images'
    VIDEO_DIR = f'{OUTPUT_DIR}/videos'
    
    # 文件名配置
    IMAGE_PREFIX = 'detect_'
    VIDEO_PREFIX = 'video_'
    
    # 视频编码和格式设置
    VIDEO_FPS = 30
    VIDEO_CODEC = 'MJPG'  # 也可以使用 'XVID'
    VIDEO_FORMAT = '.avi'
    
    # 视频保存控制
    VIDEO_SAVE_OPTIONS = {
        'ENABLED': False,        # 是否启用视频保存
        'MAX_RESOLUTION': 1920,  # 最大分辨率
        'KEEP_RATIO': True,      # 是否保持宽高比
        'FPS': 30,              # 保存帧率
        'CODEC': 'MJPG',        # 视频编码器
        'FORMAT': '.avi'        # 视频格式
    }