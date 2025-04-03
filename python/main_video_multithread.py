import os
import sys
import cv2
import time
import numpy as np
from rknnlite.api import RKNNLite
from rknnpool import rknnPoolExecutor
import argparse

import func_multithread
import config

model_path='../model/yolo11.rknn'

cap = cv2.VideoCapture('./720p60hz.mp4')

args = argparse.Namespace(
    model_path='../model/yolo11.rknn',
    target='rk3588',
    device_id=None,
    img_show=False,
    img_save=True,
    anno_json='../../../datasets/COCO/annotations/instances_val2017.json',
    img_folder='../model',
    coco_map_test=False
)

# 线程数, 增大可提高帧率
TPEs = 3
# 初始化rknn池
pool = rknnPoolExecutor(
    rknnModel=model_path,
    TPEs=TPEs,
    func=func_multithread.myFunc)

# 初始化异步所需要的帧
if (cap.isOpened()):
    for i in range(TPEs + 1):
        ret, frame = cap.read()
        if not ret:
            cap.release()
            del pool
            exit(-1)

        pool.put(frame)


frames, loopTime, initTime = 0, time.time(), time.time()

while (cap.isOpened()):
    frames += 1
    ret, frame = cap.read()
    if not ret:
        break
    pool.put(frame)
    frame, flag = pool.get()
    if flag == False:
        break
    cv2.imshow('test', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if frames % 30 == 0:
        print("30帧平均帧率:\t", 30 / (time.time() - loopTime), "帧")
        loopTime = time.time()

print("总平均帧率\t", frames / (time.time() - initTime))
# 释放cap和rknn线程池
cap.release()
cv2.destroyAllWindows()
pool.release()