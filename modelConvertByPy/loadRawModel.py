import torch
from ultralytics import YOLO  # YOLOv8 模型定义库

def load_yolov8_model(pt_model_path, use_ema=False):
    # 加载模型文件
    model_data = torch.load(pt_model_path)
    
    # 选择权重来源（模型或 EMA）
    state_dict = model_data['ema'] if use_ema else model_data['model']
    
    # 加载 YOLOv8 模型
    model = YOLO()  # YOLOv8 基础模型实例化
    model.model.load_state_dict(state_dict)  # 加载权重
    model.eval()  # 设置为评估模式
    return model

# 使用
pt_model_path = "your_model.pt"  # 替换为你的模型路径
model = load_yolov8_model(pt_model_path, use_ema=True)  # 使用 EMA 权重
