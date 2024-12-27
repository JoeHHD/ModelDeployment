import json

def get_coco_class_names_from_annotations(annotations_path):
    with open(annotations_path, 'r') as f:
        data = json.load(f)
    
    # 提取类别
    categories = data['categories']
    class_names = [category['name'] for category in categories]
    return class_names

# 使用
annotations_path = "/mnt/f/dataset/coco2017/annotations/instances_val2017.json"  # 替换为实际路径
class_names = get_coco_class_names_from_annotations(annotations_path)
print(class_names)
