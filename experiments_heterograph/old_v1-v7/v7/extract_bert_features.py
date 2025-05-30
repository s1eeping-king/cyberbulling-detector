import os
import sys
import torch
import numpy as np
from bert_feature_extractor import BertFeatureExtractor

def main():
    """运行BERT特征提取"""
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 检查GPU可用性
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 创建输出目录
    output_dir = "data/processed/bert_features"
    os.makedirs(output_dir, exist_ok=True)
    
    # 初始化BERT特征提取器
    extractor = BertFeatureExtractor(output_dim=64, device=device)
    
    # 处理评论数据
    comments_file = "data/processed/integration/comments.json"
    comments_output = os.path.join(output_dir, "comment_bert_features.json")
    
    if not os.path.exists(comments_output):
        print("开始处理评论数据...")
        extractor.process_comments(comments_file, comments_output)
    else:
        print(f"评论BERT特征文件已存在: {comments_output}")
    
    # 处理媒体会话数据
    media_file = "data/processed/integration/media_sessions.json"
    media_output = os.path.join(output_dir, "media_bert_features.json")
    
    if not os.path.exists(media_output):
        print("开始处理媒体会话数据...")
        extractor.process_media_sessions(media_file, media_output)
    else:
        print(f"媒体会话BERT特征文件已存在: {media_output}")
    
    print("BERT特征提取完成!")

if __name__ == "__main__":
    main()
