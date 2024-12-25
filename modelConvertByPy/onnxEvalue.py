import torch
import torchvision.transforms as T
from torchvision.datasets import CocoDetection
from torchvision.transforms import functional as F
import cv2
import numpy as np
from tqdm import tqdm
import os
import json

# 模型加载
def load_model(pt_model_path):
    model = torch.load(pt_model_path)  # 加载 PyTorch 模型
    model.eval()  # 设置为评估模式
    return model

# 数据预处理
def preprocess_image(image, input_size):
    transform = T.Compose([
        T.ToTensor(),
        T.Resize((input_size, input_size)),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

# 推理函数
def inference(model, image, input_size):
    transform = T.Compose([
        T.ToTensor(),
        T.Resize((input_size, input_size)),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(input_tensor)

    return outputs

# 将推理结果转换为 COCO 格式
def convert_to_coco_format(predictions, image_id, conf_threshold=0.5):
    coco_results = []
    for pred in predictions:
        boxes = pred["boxes"].cpu().numpy()
        scores = pred["scores"].cpu().numpy()
        labels = pred["labels"].cpu().numpy()

        for box, score, label in zip(boxes, scores, labels):
            if score < conf_threshold:
                continue
            x1, y1, x2, y2 = box
            coco_results.append({
                "image_id": image_id,
                "category_id": int(label),
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "score": float(score)
            })
    return coco_results

# 验证函数
def evaluate_coco(model, coco_annotations_path, image_dir, input_size=640, conf_threshold=0.5):
    # 加载 COCO 数据集
    coco_dataset = CocoDetection(
        root=image_dir, 
        annFile=coco_annotations_path, 
        transform=lambda img: F.resize(img, (input_size, input_size))
    )
    coco_results = []
    # 暂时只处理三张，验证一下模型
    print("Running inference on first 3 imgaes whithin COCO dataset...")
    cnt=0
    for image, target in tqdm(coco_dataset):
        cnt+=1
        if cnt>=3:
            break
        # 获取 image_id
        image_id = target[0]["image_id"]

        # 推理
        predictions = inference(model, image, input_size)

        # 转换为 COCO 格式
        coco_results.extend(convert_to_coco_format(predictions, image_id, conf_threshold))

    # 保存结果到临时文件
    result_file = "coco_results.json"
    with open(result_file, "w") as f:
        json.dump(coco_results, f)
    
    # 计算指标
    coco_dt = coco_dataset.coco.loadRes(result_file)
    coco_eval = coco_dataset.coco.loadRes(result_file)
    coco_eval = COCOeval(coco_dataset.coco, coco_dt, iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
# 主程序
if __name__ == "__main__":
    pt_model_path = "/root/project/ModelDeployment/raw_model/yolov8m.pt"  # 替换为你的模型路径
    coco_annotations_path = "/mnt/f/dataset/coco2017/annotations"  # 替换为 COCO annotations 路径
    image_dir = "/mnt/f/dataset/coco2017/val2017"  # 替换为 COCO 图像目录
    input_size = 640  # 模型输入大小

    # 加载模型
    # model = load_model(pt_model_path)
    # model.eval()

    model_data = torch.load(pt_model_path)  # 加载模型文件
    print(type(model_data))  # 查看数据类型
    if isinstance(model_data, dict):
        print(model_data.keys())  # 如果是字典，查看它的键

    # 评估模型
    # evaluate_coco(model, coco_annotations_path, image_dir, input_size)
