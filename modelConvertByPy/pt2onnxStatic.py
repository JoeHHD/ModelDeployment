from ultralytics import YOLO

# 加载 YOLOv8 模型
model = YOLO("/root/project/ModelDeployment/raw_model/yolov8m.pt")

# 导出 ONNX 模型，设置固定的 batch size 为 4
model.export(format="onnx", dynamic=False, batch=4)
