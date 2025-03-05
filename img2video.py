from moviepy import ImageClip, concatenate_videoclips
import os

# 图片文件列表
# image_dir = "/home/visionnav/code/sam2/dataset/ford-brighteye/test"  # 确保这些图片文件存在
# image_dir = "/home/visionnav/code/sam2/dataset/wrappallet/test"  # 确保这些图片文件存在
image_dir = "/home/visionnav/code/sam2/dataset/pallet/test"  # 确保这些图片文件存在
img_files = [
    p for p in os.listdir(image_dir)
    if os.path.splitext(p)[-1] in [".jpg"]
]

image_files = []
for filename in img_files:
    imgfilePath = os.path.join(image_dir, filename)
    image_files.append(imgfilePath)

image_files = image_files[0:200]

# 图片持续时间（秒）
duration = 0.5  # 每个图片显示的持续时间

# 创建一个视频剪辑列表
clips = [ImageClip(m).with_duration(duration) for m in image_files]

# 合并所有剪辑成一个视频
final_clip = concatenate_videoclips(clips)

# 输出视频文件
final_clip.write_videofile("./dataset/pallet.mp4", fps=24)  # fps是每秒帧数，可以根据需要调整