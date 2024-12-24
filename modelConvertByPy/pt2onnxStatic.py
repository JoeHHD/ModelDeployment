## 该文件主要用于将pt模型转成 static onnx 模型 
from ultralytics import YOLO
import os
"""
将 YOLOv8 的 .pt 模型转换为 .onnx 模型

:param pt_model_path: .pt 模型路径
:param onnx_model_path: 导出的 .onnx 模型保存路径
:param opset_version: ONNX opset 版本，默认为 11
:param dynamic: 是否使用动态尺寸支持，默认为 False
"""

def convert_to_onnx(pt_model_path, onnx_model_path, opset_version=12, dynamic=False):

    # 加载 YOLOv8 模型
    model = YOLO(pt_model_path)

    # 导出到 ONNX 格式
    # 导出到 ONNX 格式
    model.export(
        format="onnx",              # 导出格式
        opset=opset_version,        # ONNX Opset 版本
        dynamic=dynamic,            # 动态输入支持
        half=False,                 # 使用 FP32 精度
        simplify=True,              # 简化 ONNX 模型
        project=onnx_model_path,    # 自定义项目路径
        name="yolov8n_static"       # 自定义模型文件名
    )
    # print(f"Model has been exported to {onnx_model_path}")

if __name__ == "__main__":
    # 输入 .pt 模型路径
    pt_model_path = "/home/joe/project/pth2onnx/raw_model/yolov8m.pt"

    # 输出 .onnx 模型路径
    onnx_model_path = "/home/joe/project/pth2onnx/onnx_model"

    # 检查路径是否存在
    if os.path.exists(onnx_model_path):
        print("output path exist !")
        # 转换模型
        # starting from '../raw_model/yolov8n.pt' with input shape (1, 3, 640, 640) BCHW 
        # and output shape(s) (1, 84, 8400) (6.2 MB)
        convert_to_onnx(pt_model_path, onnx_model_path, opset_version=12)
    else:
        print("output path error ! !")

