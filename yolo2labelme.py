import os
import json

# 定义类别映射
class_map = {0: "Goods", 1: "Forklift", 2: "Human"}  # 根据类别定义填写，这里示例是2个类别


def yolo_to_labelme(txt_path, img_width, img_height):
    with open(txt_path, 'r') as file:
        lines = file.readlines()

    shapes = []
    for line in lines:
        parts = line.strip().split()
        class_id = int(parts[0])
        x_center = float(parts[1])
        y_center = float(parts[2])
        width = float(parts[3])
        height = float(parts[4])

        # 转换为绝对坐标
        x_center *= img_width
        y_center *= img_height
        width *= img_width
        height *= img_height

        # 计算矩形的四个顶点
        x1 = x_center - width / 2
        y1 = y_center - height / 2
        x2 = x_center + width / 2
        y2 = y_center + height / 2

        shapes.append({
            'label': class_map[class_id],
            'points': [[x1, y1], [x2, y2]],
            'group_id': None,
            'shape_type': 'rectangle',
            'flags': {}
        })

    return shapes


# 文件夹路径
txt_folder_path = "dataset/ford-brighteye"
json_output_path = "dataset/ford-brighteye"

# 图片宽度和高度（按图片尺寸读取并调整）
img_width = 1920
img_height = 1080

if not os.path.exists(json_output_path):
    os.makedirs(json_output_path)

print(txt_folder_path)
# 遍历所有txt文件并转换
for txt_file in os.listdir(txt_folder_path):
    if txt_file.endswith('.txt'):
        print(txt_file)
        txt_path = os.path.join(txt_folder_path, txt_file)
        shapes = yolo_to_labelme(txt_path, img_width, img_height)

        # 创建LabelMe格式的json文件
        labelme_data = {
            'version': '4.5.6',
            'flags': {},
            'shapes': shapes,
            'imagePath': txt_file.replace('.txt', '.jpg'),
            'imageData': None,
            'imageHeight': img_height,
            'imageWidth': img_width
        }

        json_path = os.path.join(json_output_path, txt_file.replace('.txt', '.json'))
        print(json_path)
        with open(json_path, 'w') as json_file:
            json.dump(labelme_data, json_file, indent=2)