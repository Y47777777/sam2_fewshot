'''
SAM2 少样本跨图模式：
> 1. 不限制输入图像的名称，
> 2. 将“模型加载所有图像帧数据”修改成“模型首先加载所有few-shot图像”
> 3. 给小样本图像增加prompt。
> 4. 加载待标注图像，存储到inference_state字典中。
> 5. 不再将测试图片的存储在memory bank即inference_state["output_dict"]。

支持功能：
1. 多mask prompt输入,暂时不支持point，bbox输入
2. 少样本prompt

'''
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from sam2.benchmark import model_cfg
from sam2.build_sam import build_sam2_video_predictor
import json
from tqdm import tqdm
import cv2
import time
from pycococreatortools import pycococreatortools
import argparse
import io
import base64
from skimage import measure

torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
if torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

color = [[0, 0, 255], [0, 255, 0], [255, 0, 0], [0, 255, 255], [255, 0, 255], [255, 255, 0],
         [31, 119, 180], [255, 126, 14], [44, 160, 44], [214, 39, 40],
         [147, 102, 188], [140, 86, 75], [225, 119, 194], [126, 126, 126],
         [188, 188, 34], [23, 188, 206]]


def img_tobyte(img_pil):
    '''
    该函数用于将图像转化为base64字符类型
    :param img_pil: Image类型
    :return base64_string: 字符串
    '''
    ENCODING = 'utf-8'
    img_byte = io.BytesIO()
    img_pil.save(img_byte, format='PNG')
    binary_str2 = img_byte.getvalue()
    imageData = base64.b64encode(binary_str2)
    base64_string = imageData.decode(ENCODING)
    return base64_string


def save_mask_to_json(image_path, pr_mask_dic, save_root, value2key, save_name):
    ''''
    以coco形式保存mask
    '''
    # labels_name_dic[<label_name>]=<obj_id>
    image = Image.open(image_path)
    width, height = image.size
    imageData = img_tobyte(image)
    Json_output = {
        "version": "3.16.7",
        "flags": {},
        "imagePath": {},
        "shapes": [],
        "imageData": imageData,
        "imageHeight": height,
        "imageWidth": width
    }

    # 分别对掩码中的label结果绘制边界点
    for obj_id, obj_pr_mask in pr_mask_dic.items():
        if np.max(obj_pr_mask) == 0: continue
        # 使用 measure.find_contours 来检测轮廓
        contours = measure.find_contours(obj_pr_mask, level=0.5)
        # 初始化用于存储多边形数据的列表
        polygons = []
        for contour in contours:
            # 将轮廓坐标转换为 [x1, y1, x2, y2, ..., xn, yn] 格式
            # 并且需要注意 COCO 格式中，第一个维度是 x 坐标（列），第二个是 y 坐标（行）
            contour[:, [0, 1]] = contour[:, [1, 0]]  # 交换坐标，使其为 (x, y)
            # 将坐标列表展平
            polygon = contour.ravel().tolist()
            polygons.append(polygon)

        seg_info = {'points': polygons,
                    "fill_color": None,
                    "line_color": None,
                    "label": value2key[obj_id],  # value2key[<obj_id>]=<label_name>
                    "shape_type": "polygon",
                    "flags": {}}
        Json_output["shapes"].append(seg_info)
    full_path = os.path.join(save_root, save_name.replace('.jpg', '.json'))
    with open(full_path, 'w') as output_json_file:
        json.dump(Json_output, output_json_file)


def show_org_mask(image_path, pr_mask_dic, save_root, labels_name_dic, save_name, cur_iou_dic=None):
    '''
    在原图上显示mask结果
    并保存
    '''
    img = cv2.imread(image_path)  # (726,1337,3)
    img = np.array(img, dtype=np.uint8)
    mask2img = np.zeros_like(img)  # 原图+mask
    for obj_id, obj_pr_mask in pr_mask_dic.items():
        mask_3_channels = cv2.merge([obj_pr_mask * color[obj_id][0],
                                     obj_pr_mask * color[obj_id][1],
                                     obj_pr_mask * color[obj_id][2]])
        mask2img += mask_3_channels
    img_with_masks = cv2.addWeighted(mask2img, 1, img, 1, 0)

    if cur_iou_dic:
        y = 50
        for label_name, iou in cur_iou_dic.items():
            text = label_name + ' IOU=' + str(iou)  # 左上角增加iou
            cv2.putText(img_with_masks, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            y += 50
    cv2.imwrite(os.path.join(save_root, "result_" + save_name), img_with_masks)


def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)


def get_labels(jsonfilePath):
    labels = []
    with open(jsonfilePath, "r", encoding='utf-8') as jsonf:
        jsonData = json.load(jsonf)
        for obj in jsonData["shapes"]:
            label = obj["label"]
            labels.append(label)
    return labels


def convertPolygonToMask(jsonfilePath):
    # 将json中Polygon转为mask
    mask_dic = {}
    with open(jsonfilePath, "r", encoding='utf-8') as jsonf:
        jsonData = json.load(jsonf)
        img_h = jsonData["imageHeight"]
        img_w = jsonData["imageWidth"]
        for obj in jsonData["shapes"]:
            label = obj["label"]
            polygonPoints = obj["points"]
            polygonPoints = np.array(polygonPoints, np.int32)
            mask = np.zeros((img_h, img_w), np.uint8)
            cv2.drawContours(mask, [polygonPoints], -1, (255), -1)
            mask_dic[label] = mask

    return mask_dic


def calculate_iou(pr_mask, gt_mask):
    assert pr_mask.shape == gt_mask.shape, "掩码的尺寸必须相同"
    intersection = np.logical_and(pr_mask, gt_mask).sum()
    union = np.logical_or(pr_mask, gt_mask).sum()
    iou = intersection / union
    return iou


def calculate_ap(res_dic):
    for label_name, value in res_dic.items():
        iou_list = value['iou_list']
        for thre in np.arange(0.5, 0.951, 0.05):
            sum_ = sum([1 for x in iou_list if x > thre])
            res = round(sum_ / len(iou_list), 3)
            value['recall_500595'].append(round(res, 3))
        value['mean_recall_500595'] = np.mean(value['recall_500595'])
    return res_dic


def load_test_img_as_tensor(img_path, image_size):
    img_mean = (0.485, 0.456, 0.406)
    img_std = (0.229, 0.224, 0.225)
    img_mean = torch.tensor(img_mean, dtype=torch.float32)[:, None, None]
    img_std = torch.tensor(img_std, dtype=torch.float32)[:, None, None]
    # from segment-anything-2>sam2>utils>misc.py
    img_pil = Image.open(img_path)
    img_np = np.array(img_pil.convert("RGB").resize((image_size, image_size)))
    if img_np.dtype == np.uint8:  # np.uint8 is expected for JPEG images
        img_np = img_np / 255.0
    else:
        raise RuntimeError(f"Unknown image dtype: {img_np.dtype} on {img_path}")
    img = torch.from_numpy(img_np).permute(2, 0, 1)
    width, height = img_pil.size  # the original video size
    # normalize by mean and std
    img -= img_mean
    img /= img_std
    img = img.unsqueeze(0).cuda()
    return img, height, width


def get_mask(img, json_file_path):
    # 获取图像尺寸
    height, width = img.shape[:2]
    # 创建一个空白掩码
    mask = np.zeros((height, width), dtype=np.uint8)

    with open(json_file_path) as f:
        data = json.load(f)

    # 遍历所有标注
    for shape in data['shapes']:
        if shape['label'] == 'starCirclePosition':
            # 获取points
            points = np.array(shape['points'], dtype=np.int32)
            # 填充多边形
            cv2.fillPoly(mask, [points], color=255)

    return mask


def predict(few_shot_path, test_path, save_root, few_shot_num, labels_name_dic, input_points=None, input_labels=None,
            input_bboxs=None, input_mask=None):
    """
    模型加载
    """
    sam2_checkpoint = "checkpoints/sam2_hiera_large.pt"  # 根据自己下载的模型修改
    # model_cfg = "sam2_hiera_l.yaml"  # 根据自己下载的模型修改
    model_cfg = "configs/sam2/sam2_hiera_l.yaml"
    predictor = build_sam2_video_predictor(config_file=model_cfg, ckpt_path=sam2_checkpoint)

    """
    加载模板图和mask
    """
    inference_state = predictor.init_state(video_path=few_shot_path)
    if input_labels: input_labels = np.array(input_labels, np.int32)
    if input_points or input_bboxs:  # 暂时不支持point和bboxs输入
        print('暂时不支持point和bboxs输入，请输入mask')
        return
        input_bboxs = np.array(input_bboxs, dtype=np.float32)
        input_points = np.array(input_points, dtype=np.float32)
        predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=ann_obj_id,
            points=input_points,
            labels=input_labels,
            box=input_bboxs,
        )
    elif input_mask:  # 支持mask输入
        for ann_frame_idx, masks_dic in enumerate(input_mask):
            for i, (key, mask) in enumerate(masks_dic.items()):
                ann_obj_id = labels_name_dic[key]  # labels_name_dic[<label_name>]=<obj_id>
                predictor.add_new_mask(
                    inference_state=inference_state,
                    frame_idx=ann_frame_idx,
                    obj_id=ann_obj_id,
                    mask=mask,
                )
    else:
        print('没有输入mask！请检查输入！')
        return

    '''
    信息记录
    '''
    value2key = {}
    for key, value in labels_name_dic.items():
        value2key[value] = key  # value2key[<obj_id>]=<label_name>

    # 存储结果的dic
    res = {}
    for label_name in labels_name_dic:  # labels_name_dic[<label_name>]=<obj_id>
        if label_name not in res:
            res[label_name] = {'all': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'tp': 0,
                               'iou_list': [], 'recall_500595': [], 'mean_recall_500595': 0}

    '''
    跨图分割
    '''
    fn, fp, tp, tn = 0, 0, 0, 0
    iou_list = []
    os.makedirs(save_root, exist_ok=True)
    # 加载测试数据
    for filename in tqdm(os.listdir(test_path), desc="test images:"):
        if filename.endswith(".jpg"):
            text_img_path = os.path.join(test_path, filename)
            img, height, width = load_test_img_as_tensor(text_img_path, 1024)
            inference_state["images"] = torch.cat((inference_state["images"], img), dim=0)
            inference_state["num_frames"] = few_shot_num + 1  # few_shot_num+test

            # run propagation throughout the video and collect the results in a dict
            video_segments = {}
            for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
                if out_frame_idx != few_shot_num: continue
                video_segments[out_frame_idx] = {
                    out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                    for i, out_obj_id in enumerate(out_obj_ids)
                }
            pr_mask_dic = video_segments[few_shot_num]

            # 如果有测试图像可以推理看SAM2的精度
            gt_mask_path = os.path.join(test_path, filename.replace('jpg', 'json'))
            if os.path.isfile(gt_mask_path):
                gt_mask_dic = convertPolygonToMask(gt_mask_path)

            if os.path.isfile(gt_mask_path): cur_iou_dic = {}
            for obj_id, obj_pr_mask in pr_mask_dic.items():
                obj_pr_mask = np.squeeze(np.array(obj_pr_mask, dtype=np.uint8), axis=0)
                if obj_pr_mask.shape != (height, width):
                    obj_pr_mask = cv2.resize(obj_pr_mask, (width, height))  # (w,h)
                pr_mask_dic[obj_id] = obj_pr_mask  # 更新

                # 如果有测试图像可以推理看SAM2的精度
                if os.path.isfile(gt_mask_path):
                    label_name = value2key[obj_id]
                    res[label_name]['all'] += 1
                    if label_name not in gt_mask_dic:  # gt中不存在该obj
                        if obj_pr_mask.max() == 0:
                            res[label_name]['tn'] += 1
                        else:
                            res[label_name]['fp'] += 1
                        cur_iou = 'None'
                    else:  # gt中存在该obj
                        gt_mask = gt_mask_dic[label_name]
                        cur_iou = round(calculate_iou(obj_pr_mask, gt_mask), 3)
                        if cur_iou < 0.5:
                            res[label_name]['fn'] += 1
                        else:
                            res[label_name]['tp'] += 1
                        res[label_name]['iou_list'].append(cur_iou)
                    cur_iou_dic[label_name] = cur_iou

            # 保存mask标注
            save_mask_to_json(image_path=text_img_path,
                              pr_mask_dic=pr_mask_dic,
                              save_root=save_root,
                              value2key=value2key,
                              save_name=filename)

            # 显示标注结果
            show_org_mask(image_path=text_img_path,
                          pr_mask_dic=pr_mask_dic,
                          save_root=save_root,
                          labels_name_dic=labels_name_dic,
                          save_name=filename,
                          cur_iou_dic=cur_iou_dic if os.path.isfile(gt_mask_path) else None)

            # 删除memory信息
            inference_state["images"] = inference_state["images"][:-1]
            if few_shot_num in inference_state["cached_features"]:
                del inference_state["cached_features"][few_shot_num]
            # if few_shot_num in inference_state["output_dict"]['non_cond_frame_outputs']:
            #     del inference_state["output_dict"]['non_cond_frame_outputs'][few_shot_num]
            # if few_shot_num in inference_state["output_dict_per_obj"]['non_cond_frame_outputs']:
            #     del inference_state["output_dict_per_obj"]['non_cond_frame_outputs'][few_shot_num]

    # 如果有测试图像可以推理看SAM2的精度
    if os.path.isfile(gt_mask_path):
        res = calculate_ap(res)
        for label_name, value in res.items():
            print('----', label_name, '----')
            print('all=', value['all'], 'TP=', value['tp'], 'FN=', value['fn'], 'TN=', value['tn'], 'FP=', value['fp'])
            print('mean IOU=', round(np.mean(value['iou_list'])))
            print('recall.50:.05:.95=', value['recall_500595'], '=', value['mean_recall_500595'])
        return res
    else:
        return None


def Parser():
    parser = argparse.ArgumentParser(description='timm model')
    parser.add_argument('--few_shot_path', type=str, default='/home/visionnav/code/sam2/dataset/fewshot', help='模板图和mask')
    parser.add_argument('--test_path', type=str, default='/home/visionnav/code/sam2/dataset/test', help='待打标的图像')
    parser.add_argument('--save_root', type=str, default='/home/visionnav/code/sam2/save_path', help='保存路径')
    args = parser.parse_args()
    return args


def main(args):
    few_shot_path = args.few_shot_path
    test_path = args.test_path
    save_root = args.save_root

    few_shot_json_names = [
        p for p in os.listdir(few_shot_path)
        if os.path.splitext(p)[-1] in [".json"]
    ]
    few_shot_json_names.sort()  # 不排序获取的顺序不对
    few_shot_num = len(few_shot_json_names)
    print('模板图数量：', few_shot_num)

    labels_name_dic = {}
    all_recall_500595 = [0] * 10
    mask_list = []
    for filename in few_shot_json_names:
        jsonfilePath = os.path.join(few_shot_path, filename)
        # 将json中Polygon转为mask，输出[类别名称]=mask
        masks_dic = convertPolygonToMask(jsonfilePath)
        mask_list.append(masks_dic)
        for label_name in list(masks_dic):
            if label_name not in labels_name_dic:
                labels_name_dic[label_name] = len(list(labels_name_dic))

    res_dic = predict(few_shot_path, test_path, save_root, few_shot_num, labels_name_dic, input_mask=mask_list)
    if res_dic != None:
        print('---all---')
        all_recall_500595 = []
        for key, value in res_dic.items():
            all_recall_500595.append(value['recall_500595'])
        all_recall_500595 = np.mean(all_recall_500595, axis=0).tolist()
        print('all recall.50:.05:.95=', (all_recall_500595), '=', round(np.mean(all_recall_500595), 3))


if __name__ == "__main__":
    args = Parser()
    main(args)
