import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, HeteroConv
from torch_geometric.data import HeteroData
from typing import Dict, List, Tuple
from data_loader import DataLoader
import random

class HeteroGAT(nn.Module):
    """异构图注意力网络"""
    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        # 节点特征维度
        self.feature_dims = {
            'user': 6,          # [followerCount, followingCount, likeCount, postCount, username_encoding, description_encoding]
            'media_session': 10,  # [likeCount, commentCount, loopCount, repostCount, desc_encoding, emotion_conf, theme_conf, time, emotion, theme]
            'comment': 2,       # [text_encoding, postId_hash]
            'label': 4         # [is_bullying, is_aggression, bullying_conf, aggression_conf]
        }
        
        # 边的类型
        self.edge_types = [
            ('user', 'publishes', 'media_session'),
            ('user', 'creates', 'comment'),
            ('comment', 'mentions', 'user'),
            ('media_session', 'has_label', 'label'),
            ('comment', 'belongs_to', 'media_session')
        ]
        
        # 特征转换层
        self.encoders = nn.ModuleDict({
            node_type: nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            )
            for node_type, dim in self.feature_dims.items()
        })
        
        # GAT层
        self.conv = HeteroConv({
            edge_type: GATConv(hidden_dim, hidden_dim, add_self_loops=False)
            for edge_type in self.edge_types
        })
        
        # 预测层
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 2)  # 二分类：霸凌和攻击性
        )
        
    def forward(self, x_dict, edge_index_dict):
        """前向传播"""
        # 验证边索引
        for edge_type, edge_index in edge_index_dict.items():
            src_type, _, dst_type = edge_type
            if edge_index.shape[1] > 0:  # 如果有边
                max_src_idx = edge_index[0].max().item()
                max_dst_idx = edge_index[1].max().item()
                src_nodes = x_dict[src_type].shape[0]
                dst_nodes = x_dict[dst_type].shape[0]
                
                # 验证索引是否有效
                if max_src_idx >= src_nodes or max_dst_idx >= dst_nodes:
                    # 移除无效的边
                    mask = (edge_index[0] < src_nodes) & (edge_index[1] < dst_nodes)
                    edge_index_dict[edge_type] = edge_index[:, mask]
            
        # 1. 特征编码
        h_dict = {}
        for node_type, x in x_dict.items():
            if node_type in self.encoders:
                h_dict[node_type] = self.encoders[node_type](x)
            else:
                print(f"Warning: No encoder found for {node_type}")
        
        # 2. 消息传递
        h_dict = self.conv(h_dict, edge_index_dict)
        
        # 3. 预测（仅对媒体会话节点）
        return self.classifier(h_dict['media_session'])

def calculate_metrics(outputs, labels):
    """计算分类指标"""
    predictions = (torch.sigmoid(outputs) > 0.5).float()
    
    # 计算每个类别的指标
    metrics = {}
    for i, task in enumerate(['bullying', 'aggression']):
        tp = ((predictions[:, i] == 1) & (labels[:, i] == 1)).sum().item()
        fp = ((predictions[:, i] == 1) & (labels[:, i] == 0)).sum().item()
        tn = ((predictions[:, i] == 0) & (labels[:, i] == 0)).sum().item()
        fn = ((predictions[:, i] == 0) & (labels[:, i] == 1)).sum().item()
        
        # 避免除零
        eps = 1e-7
        accuracy = (tp + tn) / (tp + tn + fp + fn + eps)
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2 * (precision * recall) / (precision + recall + eps)
        
        metrics[task] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    return metrics
