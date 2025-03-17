import os
import torch
import clip
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from PIL import Image
import cv2
import numpy as np

# 定义保存特征的函数
def save_features(features, save_path):
    if save_path.endswith(".npy"):
        np.save(save_path, features.detach().cpu().numpy())  # 使用.detach()来避免错误
    elif save_path.endswith(".pt"):
        torch.save(features, save_path)  # 保存为Torch张量
    else:
        raise ValueError("Unsupported file format. Use .npy or .pt.")

# 定义提取视频帧特征的函数
def extract_frame_features(video_path, frame_interval=1, save_dir=None, save_features_path=None):
    # 加载CLIP模型
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0

    # 创建保存目录（如果需要）
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            # 将BGR格式转换为RGB格式
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # 将NumPy数组转换为PIL图像
            frame_image = Image.fromarray(frame)
            # 保存帧为图像文件（如果需要）
            if save_dir:
                frame_image.save(os.path.join(save_dir, f"frame_{frame_count:04d}.jpg"))
            # 预处理帧
            frame_tensor = preprocess(frame_image).unsqueeze(0).to(device)
            frames.append(frame_tensor)
        frame_count += 1

    cap.release()

    # 检查 frames 列表是否为空
    if not frames:
        print(f"No frames extracted from video: {video_path}")
        return None

    # 提取视频帧的特征
    frame_features = torch.cat([model.encode_image(frame) for frame in frames])

    # 保存特征（如果需要）
    if save_features_path:
        save_features(frame_features, save_features_path)

    return frame_features

# 主函数
if __name__ == "__main__":
    # 文件路径
    text_file = "data/Vine_videos/urls_to_postids.txt"  # 包含视频URL和PostID的文本文件
    video_folder = "data/Vine_videos/videoFolder"  # 包含视频文件的文件夹
    save_dir = "data/Vine_videos/frame_images"  # 保存帧的目录
    save_features_folder = "data/Vine_videos/frame_features"  # 保存特征的文件夹
    failed_videos_file = "data/Vine_videos/failed_videos.txt"  # 记录无法读取的视频ID

    # 读取文本文件中的视频URL和PostID映射
    url_to_postid = {}
    with open(text_file, mode='r') as file:
        for line in file:
            postid, url = line.strip().split(',')
            url_to_postid[url] = postid

    # 确保保存特征的文件夹存在
    if not os.path.exists(save_features_folder):
        os.makedirs(save_features_folder)

    # 打开记录失败视频的文件
    with open(failed_videos_file, mode='w') as failed_file:
        # 遍历文本文件中的每个视频
        for url, postid in url_to_postid.items():
            video_path = os.path.join(video_folder, f"{postid}.mp4")
            if os.path.exists(video_path):
                print(f"Processing video: {video_path}")
                # 为每个视频创建独立的帧保存目录
                video_save_dir = os.path.join(save_dir, postid)
                if not os.path.exists(video_save_dir):
                    os.makedirs(video_save_dir)
                save_features_path = os.path.join(save_features_folder, f"{postid}.npy")
                frame_features = extract_frame_features(video_path, frame_interval=1, save_dir=video_save_dir, save_features_path=save_features_path)
                if frame_features is None:
                    # 如果无法提取特征，记录视频ID到文件
                    failed_file.write(f"{postid}\n")
                    print(f"Failed to extract features from video: {video_path}")
                else:
                    print(f"Extracted features shape: {frame_features.shape}")
                    print(f"Features saved to: {save_features_path}")
            else:
                print(f"Video file not found: {video_path}")