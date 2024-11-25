## 该文件主要用于验证 onnx 模型是否可以正常工作
import onnx
import onnxruntime as ort
import numpy as np

def infer_onnx_model(onnx_model_path, input_data):
    # 1. 加载 ONNX 模型并检查模型结构
    print(f"Loading ONNX model from {onnx_model_path}...")
    model = onnx.load(onnx_model_path)
    onnx.checker.check_model(model)
    print("Model structure is valid.")

    # 2. 创建 ONNX Runtime 推理会话
    session = ort.InferenceSession(onnx_model_path)
    input_name = session.get_inputs()[0].name  # 获取第一个输入的名称
    output_name = session.get_outputs()[0].name  # 获取第一个输出的名称
    print(f"Input name: {input_name}, Output name: {output_name}")

    # 3. 推理
    print("Performing inference...")
    output = session.run([output_name], {input_name: input_data})
    print("Inference completed.")

    return output[0]

if __name__ == "__main__":
    # 替换为你的 ONNX 模型路径
    onnx_model_path = "model.onnx"

    # 生成一个与模型输入匹配的随机张量
    # 假设模型输入形状为 (1, 3, 224, 224)
    input_shape = (1, 3, 224, 224)
    input_data = np.random.randn(*input_shape).astype(np.float32)

    # 推理并打印输出结果
    output = infer_onnx_model(onnx_model_path, input_data)
    print(f"Model output: {output}")
