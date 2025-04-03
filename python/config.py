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
    0,1, 2, 3, 4, 5 ]
