import cv2
import numpy as np
import os
import glob
import onnxruntime as ort
# 获取类别
def get_coco_class_names():
    return [
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
        "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
        "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
        "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
        "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
        "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
        "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
        "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
        "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
        "toothbrush"
    ]
# 数据预处理函数
def preprocess_images(image_paths, input_size=640):
    batch_data = []
    original_images = []

    for image_path in image_paths:
        image = cv2.imread(image_path)
        original_images.append(image.copy())  # 保存原始图像用于后处理
        image = cv2.resize(image, (input_size, input_size))  # 调整尺寸
        image = image.astype(np.float32) / 255.0  # 归一化
        image = np.transpose(image, (2, 0, 1))  # 转换为 (C, H, W)
        batch_data.append(image)

    batch_data = np.stack(batch_data, axis=0)  # 合并成 (batch_size, C, H, W)
    return batch_data, original_images

# 后处理函数
def postprocess_batch(outputs, conf_threshold=0.5, input_size=640, original_sizes=None):
    results_batch = []
    output = outputs[0]  # 模型输出的第一个张量

    for batch_idx, image_output in enumerate(output):  # 遍历 batch 中的每张图片
        results = []
        # 提取边界框信息和置信度
        boxes = image_output[:4, :]  # 边界框坐标 (x_center, y_center, width, height)
        objectness = image_output[4, :]  # 目标置信度
        class_probs = image_output[5:, :]  # 类别概率

        # 计算原图缩放比例
        scale_x = original_sizes[batch_idx][1] / input_size
        scale_y = original_sizes[batch_idx][0] / input_size

        # 遍历每个候选框
        for i in range(boxes.shape[1]):
            score = objectness[i]  # 目标置信度
            if score > conf_threshold:
                # 从 (x_center, y_center, width, height) 转换为 (x1, y1, x2, y2)
                x_center, y_center, width, height = boxes[:, i]
                x1 = x_center - width / 2
                y1 = y_center - height / 2
                x2 = x_center + width / 2
                y2 = y_center + height / 2

                # 恢复到原始图像尺寸
                x1, y1, x2, y2 = int(x1 * scale_x), int(y1 * scale_y), int(x2 * scale_x), int(y2 * scale_y)

                # 获取类别
                class_id = np.argmax(class_probs[:, i])
                class_score = class_probs[class_id, i]  # 类别概率
                total_score = score * class_score  # 最终置信度

                results.append({"box": (x1, y1, x2, y2), "score": total_score, "class_id": class_id})

        results_batch.append(results)

    return results_batch


# 绘制边界框函数
def draw_boxes_batch(images, results_batch, class_names, output_dir, batch_start_idx):
    for idx, (image, results) in enumerate(zip(images, results_batch)):
        for result in results:
            x1, y1, x2, y2 = result["box"]
            score = result["score"]
            class_id = result["class_id"]

            # 绘制边界框
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # 绘制标签
            label = f"{class_names[class_id]}: {score:.2f}"
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 生成输出文件名并保存图像
        output_path = os.path.join(output_dir, f"output_{batch_start_idx + idx}.jpg")
        cv2.imwrite(output_path, image)
        print(f"结果图像已保存到: {output_path}")

# 获取文件夹中的图片路径
def get_image_paths(folder_path):
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(folder_path, ext)))
    return sorted(image_paths)  # 按文件名排序

# 推理函数
def run_batch_inference(onnx_model_path, folder_path, output_dir, batch_size=4, input_size=640, conf_threshold=0.5):
    # 加载 ONNX 模型
    session = ort.InferenceSession(onnx_model_path)

    # 获取文件夹中的图片路径
    image_paths = get_image_paths(folder_path)
    print(f"检测到 {len(image_paths)} 张图片。")

    # 创建输出文件夹
    os.makedirs(output_dir, exist_ok=True)

    # 分批处理
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]  # 当前批次的图片路径
        batch_data, original_images = preprocess_images(batch_paths, input_size)

        # 推理
        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: batch_data})

        # 调试输出
        print(f"Number of outputs: {len(outputs)}")
        print(f"Shape of outputs[0]: {outputs[0].shape}")

        # 原始图像尺寸
        original_sizes = [img.shape[:2] for img in original_images]

        # 后处理
        results_batch = postprocess_batch(outputs, conf_threshold, input_size, original_sizes)

        # 类别名称（根据模型对应的数据集）
        class_names = get_coco_class_names()

        # 绘制边界框并保存结果
        draw_boxes_batch(original_images, results_batch, class_names, output_dir, i)

# 主函数
if __name__ == "__main__":
    # 参数配置
    onnx_model_path = "/root/project/ModelDeployment/onnx_model/yolov8m.onnx"  # 替换为你的 ONNX 模型路径
    folder_path = "/mnt/f/dataset/miniDataset"  # 输入图片文件夹
    output_dir = "/mnt/f/dataset/outputData"  # 输出文件夹
    batch_size = 4  # 批量大小

    # 运行批量推理
    run_batch_inference(onnx_model_path, folder_path, output_dir, batch_size)
