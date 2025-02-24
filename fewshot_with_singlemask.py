'''
SAM2视频模式：
1. 将视频转为图像帧，并将图像帧重命名为00001.jpg等数字编号
2. 模型加载所有图像帧数据
3. 对第一帧图像增加prompt（带有prompt的帧称为cond 帧）
4. 开始处理图像帧，当前帧依据cond帧和前6帧的特征&结果为判断依据，输出当前帧的结果

SAM2 few-shot跨图模式：
1. 模型加载few-shot图像，不限制输入图像的名称
2. 对每一张few-shot图像增加prompt。
3. 处理测试图像，每一张图像依据few-shot的特征&结果为判断依据，输出当前图像的结果

支持功能：
1. 每次单mask输入
2. few-shot prompt

'''
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor
# from sklearn.metrics import precision_recall_curve, auc
import cv2 as cv
import json
from tqdm import tqdm


def show_org_mask(image_path, pr_mask, save_root, save_name, text):
    img = cv2.imread(image_path)  # (726,1337,3)
    img = np.array(img, dtype=np.uint8)
    if pr_mask.shape != img.shape:
        pr_mask = cv2.resize(pr_mask, (img.shape[1], img.shape[0]))  # (w,h)
    mask_3_channels = cv2.merge([pr_mask, pr_mask * 255, pr_mask])
    img_mask = cv2.addWeighted(mask_3_channels, 0.2, img, 0.8, 0)

    # 左上角增加iou
    position = (10, 50)  # 左上角坐标，(x, y) 格式，调整数值以改变位置
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    color = (0, 0, 255)
    thickness = 3
    cv2.putText(img_mask, text, position, font, font_scale, color, thickness)
    cv2.imwrite(os.path.join(save_root, save_name), img_mask)


def show_plt_mask(mask, ax, image_path, save_root, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    # img = Image.open(image_path).convert('RGB')
    # img = img.resize((1390, 837))
    ax.imshow(mask_image)
    plt.savefig(os.path.join(save_root, image_path))
    # overlayed_image = cv2.addWeighted(img, 0.6, mask_image, 0.4, 0)
    # cv2.imwrite("overlayed_image.jpg", overlayed_image)
    # plt.imshow(np.dstack( (img, mask_image*0.5) ))
    # plt.imsave("1.png",mask_image+img)


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


def convertPolygonToMask(jsonfilePath, label_name):
    # 将json中Polygon转为mask
    with open(jsonfilePath, "r", encoding='utf-8') as jsonf:
        jsonData = json.load(jsonf)
        img_h = jsonData["imageHeight"]
        img_w = jsonData["imageWidth"]
        mask = np.zeros((img_h, img_w), np.uint8)
        # 图片中目标的数量 num=len(jsonData["shapes"])
        num = 0
        for obj in jsonData["shapes"]:
            label = obj["label"]
            if label in label_name:
                polygonPoints = obj["points"]
                polygonPoints = np.array(polygonPoints, np.int32)
                num += 1
                cv.drawContours(mask, [polygonPoints], -1, (255), -1)
        # print(num)

    return mask


# use bfloat16 for the entire notebook
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

import cv2


def calculate_iou(pr_mask, gt_mask):
    """
    计算两个掩码之间的IoU。

    参数:
        mask1: NumPy数组，包含第一个掩码,pr (736, 1337) dtype=bool
        mask2: NumPy数组，包含第二个掩码,gt (736, 1337) dtype=int

    返回:
        IoU值
    """

    # pr_mask = np.squeeze(pr_mask, axis=0)  # (837, 1390)
    # pr_mask = np.array(pr_mask, dtype=int) # pr dtype=bool->nt
    # if pr_mask.shape != gt_mask.shape:
    #     gt_mask = cv2.resize(gt_mask, (pr_mask.shape[1], pr_mask.shape[0]))
    # gt_mask = np.array(gt_mask, dtype=bool) # (736, 1337)

    # 二进制掩码数组
    assert pr_mask.shape == gt_mask.shape, "掩码的尺寸必须相同"
    intersection = np.logical_and(pr_mask, gt_mask).sum()
    union = np.logical_or(pr_mask, gt_mask).sum()
    # print('intersection=',intersection,'  union=',union)
    # print('np.sum(gt_mask)=',np.sum(gt_mask),'np.sum(pr_mask)=',np.sum(pr_mask))
    iou = intersection / union

    return iou


def calculate_ap(iou_list):
    thre_list = [0.5, 0.75, 0.9, 0.95]
    ap_0595 = []
    for thre in np.arange(0.5, 0.951, 0.05):
        sum_ = sum([1 for x in iou_list if x > thre])
        res = sum_ / len(iou_list)
        ap_0595.append(round(res, 3))
    #     print('recall',round(thre,2),'=',round(res,3))
    # print('recall 50:05:95=',np.mean(ap_0595))
    return ap_0595


def image2video(src_dir, video_dir):
    # 把图像从原始位置迁移过来，并用数字命名、且统一分辨率
    # src_dir = "/lpai/volumes/cvg-data-lx/zhangjieming/dataset/Lidata/maskrcnn_coco/valid/"
    # video_dir = "/lpai/volumes/cvg-data-lx/zhangjieming/paperwithcode/segment-anything-2/Lidata_test/weldmark_images"

    if not os.path.exists(video_dir):
        os.makedirs(video_dir)

    frame_names = [
        p for p in os.listdir(src_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
    ]
    frame_names.sort()

    i = 1
    for frame_name in frame_names:
        print(i)
        img = Image.open(os.path.join(src_dir, frame_name)).convert('RGB')
        # img = img.resize((1390, 837))
        output_name = f"{i:05d}.jpg"
        i += 1
        img.save(os.path.join(video_dir, output_name))

    # scan all the JPEG frame names in this directory
    # frame_names = [
    #     p for p in os.listdir(video_dir)
    #     if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
    # ]
    # frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

    # # take a look the first video frame
    # frame_idx = 0
    # plt.figure(figsize=(12, 8))
    # plt.title(f"frame {frame_idx}")
    # plt.imshow(Image.open(os.path.join(video_dir, frame_names[frame_idx])))


# def load_model(video_dir):
#     sam2_checkpoint = "/lpai/volumes/cvg-data-lx/zhangjieming/paperwithcode/segment-anything-2/checkpoints/sam2_hiera_large.pt"
#     model_cfg = "sam2_hiera_l.yaml"
#     predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)
#     inference_state = predictor.init_state(video_path=video_dir)
#     return predictor,inference_state

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


def predict(few_shot_path, test_path, save_root, label, few_shot_num, input_points=None, input_labels=None,
            input_bboxs=None, input_mask=None):
    sam2_checkpoint = "/lpai/volumes/cvg-data-lx/zhangjieming/paperwithcode/segment-anything-2/checkpoints/sam2_hiera_large.pt"
    model_cfg = "sam2_hiera_l.yaml"
    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)
    # 只加载few-shot
    inference_state = predictor.init_state(video_path=few_shot_path)

    ann_frame_idx = 0  # the frame index we interact with
    ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)

    # Let's add a positive click at (x, y) = (210, 350) to get started
    # for labels, `1` means positive click and `0` means negative click
    if input_points: input_points = np.array(input_points, dtype=np.float32)
    if input_labels: input_labels = np.array(input_labels, np.int32)
    if input_bboxs: input_bboxs = np.array(input_bboxs, dtype=np.float32)
    if input_points or input_bboxs:  # point和bboxs暂时不能用
        predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=ann_obj_id,
            points=input_points,
            labels=input_labels,
            box=input_bboxs,
        )
    elif input_mask:
        for ann_frame_idx, mask in enumerate(input_mask):
            predictor.add_new_mask(
                inference_state=inference_state,
                frame_idx=ann_frame_idx,
                obj_id=ann_obj_id,
                mask=mask,
            )
    '''
    few_shot帧信息已经存在cached_features
    每帧信息包括:
    image:[1,3,1024,1024]
    {"vision_features": src, # [1,256,64,64]
    "vision_pos_enc": pos, # [1,256,256,256],[1,256,128,128],[1,256,64,64]
    "backbone_fpn": features,# [1,256,256,256],[1,256,128,128],[1,256,64,64]}
    '''

    fn, fp, tp, tn = 0, 0, 0, 0
    iou_list = []
    os.makedirs(save_root, exist_ok=True)
    # 加载测试数据
    for filename in tqdm(os.listdir(test_path), desc="test images:"):
        # for filename in os.listdir(test_path):
        if filename.endswith(".jpg"):
            text_img_path = os.path.join(test_path, filename)
            img, height, width = load_test_img_as_tensor(text_img_path, 1024)
            inference_state["images"] = torch.cat((inference_state["images"], img), dim=0)
            inference_state["num_frames"] = few_shot_num + 1  # few_shot_num+test

            # run propagation throughout the video and collect the results in a dict
            video_segments = {}  # video_segments contains the per-frame segmentation results
            # predictor.propagate_in_video(inference_state)
            for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
                if out_frame_idx != few_shot_num: continue
                video_segments[out_frame_idx] = {
                    out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                    for i, out_obj_id in enumerate(out_obj_ids)
                }
            # print('len(video_segments)=',len(video_segments))

            gt_mask_path = os.path.join(test_path, filename.replace('jpg', 'json'))
            gt_mask = convertPolygonToMask(gt_mask_path, label_name=[label])
            gt_max = gt_mask.max()
            # for i in range(len(video_segments)):
            # out_mask =video_segments[i][1]
            out_mask = video_segments[few_shot_num][1]
            out_mask = np.squeeze(np.array(out_mask, dtype=np.uint8), axis=0)
            out_max = out_mask.max()
            if out_mask.shape != (height, width):
                out_mask = cv2.resize(out_mask, (width, height))  # (w,h)
            if gt_max == 0:  # gt中不存在mask
                cur_iou = 'None'
                if out_max == 0:
                    tn += 1
                else:
                    fp += 1
            else:  # gt中存在mask
                cur_iou = calculate_iou(out_mask, gt_mask)
                if cur_iou < 0.5:
                    fn += 1
                else:
                    tp += 1
                iou_list.append(cur_iou)
            if cur_iou == 'None':
                text = label + " IOU=None"
            else:
                text = label + " IOU=" + str(round(cur_iou, 3))
            show_org_mask(image_path=text_img_path,
                          pr_mask=out_mask, save_root=save_root,
                          save_name=filename,
                          text=text)
            inference_state["images"] = inference_state["images"][:-1]
            if few_shot_num in inference_state["cached_features"]:
                del inference_state["cached_features"][few_shot_num]
            # if few_shot_num in inference_state["output_dict"]['non_cond_frame_outputs']:
            #     del inference_state["output_dict"]['non_cond_frame_outputs'][few_shot_num]
            # if few_shot_num in inference_state["output_dict_per_obj"]['non_cond_frame_outputs']:
            #     del inference_state["output_dict_per_obj"]['non_cond_frame_outputs'][few_shot_num]

    ap_0595 = calculate_ap(iou_list)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    all = fn + fp + tp + tn
    mean_iou = np.mean(iou_list)
    count_iou_0 = sum([1 for i in iou_list if i == 0.0])
    print('gt mask>0有', len(iou_list), 'pr mask=0', count_iou_0)
    print('mean IOU=', str(round(mean_iou, 3)))
    print('all=', all, 'TP=', tp, 'FN=', fn, 'TN=', tn, 'FP=', fp)
    return ap_0595


if __name__ == "__main__":
    # sam2处理帧图像，需要先将图像name转为数值编号，
    # src_dir = "/lpai/volumes/cvg-data-lx/zhangjieming/dataset/Lidata/maskrcnn_coco/valid/"
    # video_dir = "/lpai/volumes/cvg-data-lx/zhangjieming/dataset/Lidata_forsam2/Laser_welding"
    # image2video(src_dir,video_dir)

    few_shot_path = '/lpai/volumes/cvg-data-lx/zhangjieming/dataset/Lidata_forsam2/light/few_shot_org/'
    test_path = '/lpai/volumes/cvg-data-lx/zhangjieming/dataset/Lidata_forsam2/light/test_imgs/'
    save_root = '/lpai/volumes/cvg-data-lx/zhangjieming/paperwithcode/segment-anything-2/Lidata_test/light/test_imgs/'

    few_shot_json_names = [
        p for p in os.listdir(few_shot_path)
        if os.path.splitext(p)[-1] in [".json"]
    ]
    few_shot_json_names.sort()  # 不排序获取的顺序不对
    few_shot_num = len(few_shot_json_names)
    print('few_shot_num=', few_shot_num)
    jsonfilePath = os.path.join(few_shot_path, few_shot_json_names[0])
    labels = get_labels(jsonfilePath)
    labels = ['starCirclePosition']
    all_ap_0595 = [0] * 10
    for label in labels:
        print(label, 'doing...')
        mask_list = []
        for filename in few_shot_json_names:
            jsonfilePath = os.path.join(few_shot_path, filename)
            mask = convertPolygonToMask(jsonfilePath, label)
            mask_list.append(mask)
        ap_0595 = predict(few_shot_path, test_path, save_root, label, few_shot_num, input_mask=mask_list)
        print('recall.50:.05:.95=', ap_0595, '=', np.mean(ap_0595))

        for i in range(len(all_ap_0595)):
            all_ap_0595[i] = (all_ap_0595[i] + ap_0595[i])

    print('all')
    for i in range(len(all_ap_0595)):
        all_ap_0595[i] = round(all_ap_0595[i] / len(labels), 3)
    print('all recall.50:.05:.95=', (all_ap_0595), '=', np.mean(all_ap_0595))
