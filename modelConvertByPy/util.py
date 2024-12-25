import onnxruntime as ort
import numpy as np

# 加载模型
session = ort.InferenceSession("/root/project/ModelDeployment/onnx_model/yolov8m.onnx")

# 创建随机输入数据（根据模型要求）
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
dummy_input = np.random.rand(1, 3, 640, 640).astype(np.float32)

# 推理
outputs = session.run([output_name], {input_name: dummy_input})

# 查看输出信息
output_data = outputs[0]  # 输出张量
print(f"Output Shape: {output_data.shape}")  # [1, 84, 8400]

# 检查第一个候选框的信息
first_box = output_data[0, :, 0]  # 获取第一个框的 84 个特征
print(f"First Box Features: {first_box}")

# 提取置信度
confidence = first_box[4]  # 索引 4 通常是 confidence
print(f"Confidence: {confidence}")
