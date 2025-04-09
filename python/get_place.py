import cv2
import numpy as np

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        # 在图片副本上绘制坐标信息，以保持原图不变
        img_copy = img.copy()
        text = f"Coordinates: ({x}, {y})"
        cv2.putText(img_copy, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (0, 255, 0), 2)
        cv2.imshow('image', img_copy)

# 读取图片（这里需要替换为实际的图片路径）
img = cv2.imread('E:\src\yolo11\output\image\detect_20250409_205744.jpg')
if img is None:
    print("错误：无法读取图片")
    exit()

# 创建窗口并设置鼠标回调
cv2.namedWindow('image')
cv2.setMouseCallback('image', mouse_callback)

# 显示图片
cv2.imshow('image', img)

# 等待按下 'q' 键退出
while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
