## yolo11.rknn and yolo11.onnx
> 官方给的模型：同样是9输出，并且除了class的数量不一样，其他都一样，而且是int8类型,onnx为float32

## best8.pt
> 418晚上训练较好的一轮模型

## https://github.com/airockchip/ultralytics_yolo11/blob/main/RKOPT_README.zh-CN.md：地址1
> pt转onnx的readme

## https://github.com/airockchip/rknn_model_zoo/blob/main/examples/yolo11/README.md：地址2
> onnx转rknn的readme

## Annotation以及它的zip
> 马一萌当时的所有出错的balck  ball的文件

## best8.onnx
> 根据best8.pt和地址1转换过来的

## best8.rknn
> 根据best8.onnx和地址2转换过来的

## fix_label.py
> 修复马一萌xml文件的代码


## 4.18
使用以上步骤转换模型切记使用fp
后续优化在于两个方面：
    1. yolo检测的帧率
    2. cpp的优势
    3. 单片机的东西得准备

