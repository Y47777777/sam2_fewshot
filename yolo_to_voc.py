import os
import xml.etree.ElementTree as ET
from xml.dom.minidom import parseString
from tkinter import Tk
from tkinter.filedialog import askdirectory

# 类别索引到名称的映射
class_mapping = {0: "Goods", 1: "Forklift", 2: "Human"}  # 修改为你的实际类别

def create_voc_xml(image_name, image_width, image_height, yolo_annotations, output_folder):
    # 创建 Pascal VOC XML 根节点
    annotation = ET.Element("annotation")
    folder = ET.SubElement(annotation, "folder")
    folder.text = "dataset"
    filename = ET.SubElement(annotation, "filename")
    filename.text = image_name
    path = ET.SubElement(annotation, "path")
    path.text = os.path.join(output_folder, image_name)

    source = ET.SubElement(annotation, "source")
    database = ET.SubElement(source, "database")
    database.text = "Unknown"

    size = ET.SubElement(annotation, "size")
    width = ET.SubElement(size, "width")
    width.text = str(image_width)
    height = ET.SubElement(size, "height")
    height.text = str(image_height)
    depth = ET.SubElement(size, "depth")
    depth.text = "3"

    segmented = ET.SubElement(annotation, "segmented")
    segmented.text = "0"

    for yolo_annotation in yolo_annotations:  # 避免与根节点变量冲突
        class_id, center_x, center_y, box_width, box_height = map(float, yolo_annotation)
        class_name = class_mapping[int(class_id)]
        xmin = int((center_x - box_width / 2) * image_width)
        ymin = int((center_y - box_height / 2) * image_height)
        xmax = int((center_x + box_width / 2) * image_width)
        ymax = int((center_y + box_height / 2) * image_height)

        obj = ET.SubElement(annotation, "object")
        name = ET.SubElement(obj, "name")
        name.text = class_name
        pose = ET.SubElement(obj, "pose")
        pose.text = "Unspecified"
        truncated = ET.SubElement(obj, "truncated")
        truncated.text = "0"
        difficult = ET.SubElement(obj, "difficult")
        difficult.text = "0"
        bndbox = ET.SubElement(obj, "bndbox")
        ET.SubElement(bndbox, "xmin").text = str(xmin)
        ET.SubElement(bndbox, "ymin").text = str(ymin)
        ET.SubElement(bndbox, "xmax").text = str(xmax)
        ET.SubElement(bndbox, "ymax").text = str(ymax)

    # 格式化 XML
    xml_string = ET.tostring(annotation, "utf-8")
    dom = parseString(xml_string)
    pretty_xml = dom.toprettyxml(indent="  ")

    # 保存 XML 文件
    xml_file_path = os.path.join(output_folder, os.path.splitext(image_name)[0] + ".xml")
    with open(xml_file_path, "w") as f:
        f.write(pretty_xml)

    print(f"Converted: {image_name} -> {xml_file_path}")

def main():
    # 文件选择对话框
    print("请选择存放 YOLO 标签的文件夹：")
    Tk().withdraw()  # 隐藏主窗口
    yolo_labels_folder = askdirectory(title="选择 YOLO 标签文件夹")
    if not yolo_labels_folder:
        print("未选择 YOLO 标签文件夹，程序退出！")
        return

    print("请选择存放图片的文件夹：")
    images_folder = askdirectory(title="选择图片文件夹")
    if not images_folder:
        print("未选择图片文件夹，程序退出！")
        return

    output_folder = "./VOC_Annotations"  # Pascal VOC 格式标签保存目录

    # 创建输出目录
    os.makedirs(output_folder, exist_ok=True)

    # 假设所有图片大小相同
    image_width, image_height = 1280, 1024  # 替换为你的图片分辨率

    # 遍历 YOLO 标签文件
    for txt_file in os.listdir(yolo_labels_folder):
        if not txt_file.endswith(".txt"):
            continue

        yolo_file_path = os.path.join(yolo_labels_folder, txt_file)
        image_name = txt_file.replace(".txt", ".jpg")

        # 读取 YOLO 标签
        with open(yolo_file_path, "r") as f:
            yolo_annotations = [line.strip().split() for line in f.readlines()]

        # 创建 Pascal VOC XML
        create_voc_xml(image_name, image_width, image_height, yolo_annotations, output_folder)

    print("转换完成！所有 Pascal VOC 标签已保存到:", output_folder)

# 特殊的入口语句
if __name__ == "__main__":
    main()
