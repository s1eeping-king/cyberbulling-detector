import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
import logging
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.data import Data as GraphData

logger = logging.getLogger(__name__)

class MultimodalAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int = 8):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads)
        
    def forward(self, text: torch.Tensor, video: torch.Tensor) -> torch.Tensor:
        # text和video的形状都是[batch_size, hidden_dim]
        # 打印输入维度用于调试
        logger.debug(f"Attention input shapes - text: {text.shape}, video: {video.shape}")
        
        # 需要调整为[seq_len, batch_size, hidden_dim]
        text = text.unsqueeze(0)  # [1, batch_size, hidden_dim]
        video = video.unsqueeze(0)  # [1, batch_size, hidden_dim]
        
        # 打印转换后的维度用于调试
        logger.debug(f"Attention adjusted shapes - text: {text.shape}, video: {video.shape}")
        
        # 计算注意力，query=text, key=video, value=video
        attn_output, _ = self.attention(query=text, key=video, value=video)
        return attn_output.squeeze(0)  # [batch_size, hidden_dim]

class MultimodalFusion(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
    def forward(self, text: torch.Tensor, video: torch.Tensor) -> torch.Tensor:
        # text和video的形状都是[batch_size, hidden_dim]
        # 连接特征
        combined = torch.cat([text, video], dim=1)  # [batch_size, hidden_dim * 2]
        # 融合
        return self.fusion(combined)  # [batch_size, hidden_dim]

class GraphNeuralNetwork(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, out_channels)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # 第一层：图卷积
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.2, training=self.training)
        
        # 第二层：图卷积
        x = F.elu(self.conv2(x, edge_index))
        x = F.dropout(x, p=0.2, training=self.training)
        
        # 第三层：图卷积
        x = self.conv3(x, edge_index)
        return x

class CyberbullyingDetector(nn.Module):
    def __init__(
        self,
        text_dim: int = 768,  # BERT特征维度
        video_dim: int = 512,  # 视频特征维度
        hidden_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.3,
        node_feature_dim: int = 64,  # 图谱节点特征维度
        gnn_hidden_dim: int = 128,   # GNN隐藏层维度
        gnn_output_dim: int = 256    # GNN输出维度
    ):
        super().__init__()
        
        # 文本特征编码器
        self.text_encoder = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 视频特征编码器
        self.video_encoder = nn.Sequential(
            nn.Linear(video_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 多模态注意力
        self.attention = MultimodalAttention(hidden_dim, num_heads)
        
        # 特征融合
        self.fusion = MultimodalFusion(hidden_dim)
        
        # 图神经网络
        self.gnn = GraphNeuralNetwork(
            in_channels=node_feature_dim,
            hidden_channels=gnn_hidden_dim,
            out_channels=gnn_output_dim
        )
        
        # 最终特征融合层
        self.final_fusion = nn.Sequential(
            nn.Linear(hidden_dim + gnn_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 分类头
        self.bullying_classifier = nn.Sequential(
            nn.Linear(hidden_dim, 2),
            nn.LogSoftmax(dim=1)
        )
        
        self.aggression_classifier = nn.Sequential(
            nn.Linear(hidden_dim, 2),
            nn.LogSoftmax(dim=1)
        )
        
    def forward(self, batch: Dict) -> Dict[str, torch.Tensor]:
        # 获取输入
        text_features = batch['text_features']  # [batch_size, text_dim]
        video_features = batch['video_features']  # [batch_size, video_dim]
        x = batch['x']  # 节点特征
        edge_index = batch['edge_index']  # 边索引
        batch_indices = batch['batch']  # batch索引
        
        # 打印输入维度用于调试
        logger.debug(f"Input shapes - text: {text_features.shape}, video: {video_features.shape}")
        
        # 确保特征维度正确
        if text_features.dim() == 1:
            text_features = text_features.unsqueeze(0)
        if video_features.dim() == 1:
            video_features = video_features.unsqueeze(0)
            
        # 编码特征
        text_encoded = self.text_encoder(text_features)  # [batch_size, hidden_dim]
        video_encoded = self.video_encoder(video_features)  # [batch_size, hidden_dim]
        
        # 注意力机制
        text_attended = self.attention(text_encoded, video_encoded)  # [batch_size, hidden_dim]
        
        # 特征融合
        fused_features = self.fusion(text_attended, video_encoded)  # [batch_size, hidden_dim]
        
        # 图神经网络处理
        gnn_features = self.gnn(x, edge_index)  # [num_nodes, gnn_output_dim]
        
        # 获取当前batch中相关节点的GNN特征
        batch_size = batch_indices.max().item() + 1
        batch_gnn_features = []
        
        for i in range(batch_size):
            # 获取当前batch中的节点索引
            node_mask = batch_indices == i
            # 获取这些节点的GNN特征
            batch_node_features = gnn_features[node_mask]  # [num_nodes_in_batch, gnn_output_dim]
            # 对节点特征进行平均池化
            batch_gnn_features.append(batch_node_features.mean(dim=0))  # [gnn_output_dim]
        
        # 堆叠所有batch的特征
        batch_gnn_features = torch.stack(batch_gnn_features)  # [batch_size, gnn_output_dim]
        
        # 将GNN特征与多模态特征融合
        final_features = self.final_fusion(
            torch.cat([fused_features, batch_gnn_features], dim=1)
        )
        
        # 分类
        bullying_logits = self.bullying_classifier(final_features)  # [batch_size, 2]
        aggression_logits = self.aggression_classifier(final_features)  # [batch_size, 2]
        
        return {
            'bullying_logits': bullying_logits,
            'aggression_logits': aggression_logits,
            # 添加中间特征
            'text_encoded': text_encoded,
            'video_encoded': video_encoded,
            'text_attended': text_attended,
            'fused_features': fused_features,
            'gnn_features': batch_gnn_features,  # 使用batch级别的GNN特征
            'final_features': final_features,
            # 添加原始特征用于分析
            'text_features': text_features,
            'video_features': video_features
        }
    
    def compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Dict
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        # 获取标签
        bullying_labels = batch['bullying_label'].squeeze()
        aggression_labels = batch['aggression_label'].squeeze()
        
        # 计算损失
        bullying_loss = F.nll_loss(outputs['bullying_logits'], bullying_labels)
        aggression_loss = F.nll_loss(outputs['aggression_logits'], aggression_labels)
        
        # 总损失
        total_loss = bullying_loss + aggression_loss
        
        # 计算准确率
        bullying_preds = outputs['bullying_logits'].argmax(dim=1)
        aggression_preds = outputs['aggression_logits'].argmax(dim=1)
        
        bullying_acc = (bullying_preds == bullying_labels).float().mean()
        aggression_acc = (aggression_preds == aggression_labels).float().mean()
        
        metrics = {
            'bullying_loss': bullying_loss.item(),
            'aggression_loss': aggression_loss.item(),
            'bullying_acc': bullying_acc.item(),
            'aggression_acc': aggression_acc.item()
        }
        
        return total_loss, metrics 