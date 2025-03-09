import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)

class MultimodalFusion(nn.Module):
    """
    多模态特征融合模块
    支持更复杂的融合方法，如注意力机制
    """
    def __init__(self, video_dim, text_dim, output_dim=512):
        """
        初始化多模态融合模块
        
        参数:
            video_dim: 视频特征维度
            text_dim: 文本特征维度
            output_dim: 输出特征维度
        """
        super().__init__()
        self.video_dim = video_dim
        self.text_dim = text_dim
        self.output_dim = output_dim
        
        # 特征映射层
        self.video_projection = nn.Linear(video_dim, output_dim)
        self.text_projection = nn.Linear(text_dim, output_dim)
        
        # 注意力层
        self.attention = nn.Sequential(
            nn.Linear(output_dim * 2, output_dim),
            nn.Tanh(),
            nn.Linear(output_dim, 2),
            nn.Softmax(dim=1)
        )
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(output_dim * 2, output_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        logger.info(f"Initialized MultimodalFusion with video_dim={video_dim}, text_dim={text_dim}, output_dim={output_dim}")
    
    def forward(self, video_features, text_features):
        """
        前向传播
        
        参数:
            video_features: 视频特征 [batch_size, video_dim]
            text_features: 文本特征 [batch_size, text_dim]
        
        返回:
            融合特征 [batch_size, output_dim]
        """
        # 特征映射
        video_proj = self.video_projection(video_features)
        text_proj = self.text_projection(text_features)
        
        # 拼接特征
        concat_features = torch.cat([video_proj, text_proj], dim=1)
        
        # 计算注意力权重
        attention_weights = self.attention(concat_features)
        
        # 加权融合
        weighted_video = video_proj * attention_weights[:, 0].unsqueeze(1)
        weighted_text = text_proj * attention_weights[:, 1].unsqueeze(1)
        
        # 拼接加权特征
        weighted_concat = torch.cat([weighted_video, weighted_text], dim=1)
        
        # 融合
        fused_features = self.fusion(weighted_concat)
        
        return fused_features
