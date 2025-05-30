import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from typing import Dict, List, Tuple, Optional

class ThreePartGAT(nn.Module):
    """
    将异构图分为三个同构部分的GAT模型：
    1. 媒体会话节点
    2. 评论图（评论节点之间有OFFENSIVE_MENTION和NON_OFFENSIVE_MENTION边）
    3. 用户图（用户节点之间有OFFENSIVE_COMMENT和NON_OFFENSIVE_COMMENT边）
    """
    def __init__(self, hidden_dim: int = 16, num_layers: int = 2, dropout: float = 0.5, heads: int = 4):
        super().__init__()
        # 节点特征维度
        self.feature_dims = {
            'user': 5,           # 5(description_offensive, followerCount_normalized, followingCount_normalized, likeCount_normalized, postCount_normalized)
            'media_session': 74, # 10(原始特征) + 64(BERT特征)
            'comment': 66,       # 2(原始特征: offensive, confidence) + 64(BERT特征)
        }

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.heads = heads

        # 用户图的GAT层
        self.user_gat_layers = nn.ModuleList()
        for i in range(num_layers):
            in_channels = self.feature_dims['user'] if i == 0 else hidden_dim * heads
            self.user_gat_layers.append(
                GATConv(
                    in_channels=in_channels,
                    out_channels=hidden_dim,
                    heads=heads,
                    dropout=dropout,
                    concat=True if i < num_layers - 1 else False
                )
            )

        # 评论图的GAT层
        self.comment_gat_layers = nn.ModuleList()
        for i in range(num_layers):
            in_channels = self.feature_dims['comment'] if i == 0 else hidden_dim * heads
            self.comment_gat_layers.append(
                GATConv(
                    in_channels=in_channels,
                    out_channels=hidden_dim,
                    heads=heads,
                    dropout=dropout,
                    concat=True if i < num_layers - 1 else False
                )
            )

        # 媒体会话节点的线性投影层
        self.media_projection = nn.Linear(self.feature_dims['media_session'], hidden_dim)

        # 最终分类器
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),  # 3倍hidden_dim是因为我们拼接了三个部分的表征
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2)  # 二分类：[非霸凌, 霸凌]
        )

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """初始化模型权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x_dict, edge_index_dict, edge_attr_dict=None):
        """前向传播"""
        # 检查是否有空的节点特征
        for node_type, x in x_dict.items():
            if x.shape[0] == 0:
                feature_dim = self.feature_dims.get(node_type, self.hidden_dim)
                x_dict[node_type] = torch.zeros((1, feature_dim), device=x.device)

        # 1. 处理用户图
        user_representation = self._process_user_graph(x_dict, edge_index_dict)

        # 2. 处理评论图
        comment_representation = self._process_comment_graph(x_dict, edge_index_dict)

        # 3. 获取媒体会话表征
        media_representation = self._get_media_representation(x_dict, edge_index_dict)

        # 5. 拼接三个部分的表征
        batch_size = edge_index_dict.get('batch_size', 1)

        # 确保所有表征都有正确的批次大小
        if user_representation.size(0) != batch_size:
            user_representation = self._adjust_batch_size(user_representation, batch_size)

        if comment_representation.size(0) != batch_size:
            comment_representation = self._adjust_batch_size(comment_representation, batch_size)

        if media_representation.size(0) != batch_size:
            media_representation = self._adjust_batch_size(media_representation, batch_size)

        # 拼接三个部分的表征
        combined_representation = torch.cat([
            user_representation,
            comment_representation,
            media_representation
        ], dim=1)

        # 6. 应用分类器
        logits = self.classifier(combined_representation)

        return logits

    def _adjust_batch_size(self, representation, target_batch_size):
        """调整表征的批次大小"""
        current_batch_size = representation.size(0)

        if current_batch_size < target_batch_size:
            # 如果当前批次小于目标批次，复制最后一个样本
            padding = representation[-1].unsqueeze(0).repeat(target_batch_size - current_batch_size, 1)
            representation = torch.cat([representation, padding], dim=0)
        elif current_batch_size > target_batch_size:
            # 如果当前批次大于目标批次，截断
            representation = representation[:target_batch_size]

        return representation

    def _process_user_graph(self, x_dict, edge_index_dict):
        """处理用户图，应用GAT层并进行平均池化"""
        if 'user' not in x_dict or x_dict['user'].size(0) == 0:
            # 如果没有用户节点，返回零向量
            return torch.zeros((1, self.hidden_dim), device=next(iter(x_dict.values())).device)

        # 获取用户节点特征
        x = x_dict['user']

        # 获取用户间的边索引
        offensive_edge_index = edge_index_dict.get(('user', 'offensive_comment', 'user'), None)
        non_offensive_edge_index = edge_index_dict.get(('user', 'non_offensive_comment', 'user'), None)

        # 合并两种边
        edge_indices = []
        if offensive_edge_index is not None and offensive_edge_index.size(1) > 0:
            edge_indices.append(offensive_edge_index)
        if non_offensive_edge_index is not None and non_offensive_edge_index.size(1) > 0:
            edge_indices.append(non_offensive_edge_index)

        if not edge_indices:
            # 如果没有边，直接返回平均池化的结果
            if 'batch_dict' in edge_index_dict and 'user' in edge_index_dict['batch_dict']:
                # 使用批次信息进行池化
                batch_indices = edge_index_dict['batch_dict']['user']
                batch_size = batch_indices.max().item() + 1

                # 对每个子图分别进行平均池化
                pooled_features = []
                for batch_idx in range(batch_size):
                    mask = (batch_indices == batch_idx)
                    if mask.sum() > 0:
                        pooled_features.append(x[mask].mean(dim=0))
                    else:
                        pooled_features.append(torch.zeros_like(x[0]))

                return torch.stack(pooled_features)
            else:
                # 如果没有批次信息，返回所有节点的平均值
                return x.mean(dim=0, keepdim=True)

        # 合并所有边索引
        edge_index = torch.cat(edge_indices, dim=1)

        # 应用GAT层
        for layer in self.user_gat_layers:
            x = layer(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # 使用批次信息进行池化
        if 'batch_dict' in edge_index_dict and 'user' in edge_index_dict['batch_dict']:
            batch_indices = edge_index_dict['batch_dict']['user']
            batch_size = batch_indices.max().item() + 1

            # 对每个子图分别进行平均池化
            pooled_features = []
            for batch_idx in range(batch_size):
                mask = (batch_indices == batch_idx)
                if mask.sum() > 0:
                    pooled_features.append(x[mask].mean(dim=0))
                else:
                    pooled_features.append(torch.zeros_like(x[0]))

            return torch.stack(pooled_features)
        else:
            # 如果没有批次信息，返回所有节点的平均值
            return x.mean(dim=0, keepdim=True)

    def _process_comment_graph(self, x_dict, edge_index_dict):
        """处理评论图，应用GAT层并进行平均池化"""
        if 'comment' not in x_dict or x_dict['comment'].size(0) == 0:
            # 如果没有评论节点，返回零向量
            return torch.zeros((1, self.hidden_dim), device=next(iter(x_dict.values())).device)

        # 获取评论节点特征
        x = x_dict['comment']

        # 获取评论间的边索引
        offensive_edge_index = edge_index_dict.get(('comment', 'OFFENSIVE_MENTION', 'comment'), None)
        non_offensive_edge_index = edge_index_dict.get(('comment', 'NON_OFFENSIVE_MENTION', 'comment'), None)

        # 合并两种边
        edge_indices = []
        if offensive_edge_index is not None and offensive_edge_index.size(1) > 0:
            edge_indices.append(offensive_edge_index)
        if non_offensive_edge_index is not None and non_offensive_edge_index.size(1) > 0:
            edge_indices.append(non_offensive_edge_index)

        if not edge_indices:
            # 如果没有边，直接返回平均池化的结果
            if 'batch_dict' in edge_index_dict and 'comment' in edge_index_dict['batch_dict']:
                # 使用批次信息进行池化
                batch_indices = edge_index_dict['batch_dict']['comment']
                batch_size = batch_indices.max().item() + 1

                # 对每个子图分别进行平均池化
                pooled_features = []
                for batch_idx in range(batch_size):
                    mask = (batch_indices == batch_idx)
                    if mask.sum() > 0:
                        pooled_features.append(x[mask].mean(dim=0))
                    else:
                        pooled_features.append(torch.zeros_like(x[0]))

                return torch.stack(pooled_features)
            else:
                # 如果没有批次信息，返回所有节点的平均值
                return x.mean(dim=0, keepdim=True)

        # 合并所有边索引
        edge_index = torch.cat(edge_indices, dim=1)

        # 应用GAT层
        for layer in self.comment_gat_layers:
            x = layer(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # 使用批次信息进行池化
        if 'batch_dict' in edge_index_dict and 'comment' in edge_index_dict['batch_dict']:
            batch_indices = edge_index_dict['batch_dict']['comment']
            batch_size = batch_indices.max().item() + 1

            # 对每个子图分别进行平均池化
            pooled_features = []
            for batch_idx in range(batch_size):
                mask = (batch_indices == batch_idx)
                if mask.sum() > 0:
                    pooled_features.append(x[mask].mean(dim=0))
                else:
                    pooled_features.append(torch.zeros_like(x[0]))

            return torch.stack(pooled_features)
        else:
            # 如果没有批次信息，返回所有节点的平均值
            return x.mean(dim=0, keepdim=True)

    def _get_media_representation(self, x_dict, edge_index_dict):
        """获取媒体会话节点的表征"""
        if 'media_session' not in x_dict or x_dict['media_session'].size(0) == 0:
            # 如果没有媒体会话节点，返回零向量
            return torch.zeros((1, self.hidden_dim), device=next(iter(x_dict.values())).device)

        # 获取媒体会话节点特征
        media_features = x_dict['media_session']

        # 使用线性层将媒体会话特征转换为hidden_dim维度
        media_features = self.media_projection(media_features)

        # 使用批次信息
        if 'batch_dict' in edge_index_dict and 'media_session' in edge_index_dict['batch_dict']:
            batch_indices = edge_index_dict['batch_dict']['media_session']
            batch_size = batch_indices.max().item() + 1

            # 对每个子图分别获取媒体会话节点
            media_representations = []
            for batch_idx in range(batch_size):
                mask = (batch_indices == batch_idx)
                if mask.sum() > 0:
                    # 每个子图只取第一个媒体会话节点（中心节点）
                    media_representations.append(media_features[mask][0])
                else:
                    media_representations.append(torch.zeros_like(media_features[0]))

            return torch.stack(media_representations)
        else:
            # 如果没有批次信息，返回第一个媒体会话节点
            return media_features[0].unsqueeze(0)

def calculate_metrics(outputs, labels):
    """计算二分类指标（霸凌/非霸凌）"""
    # 将输出转换为概率
    probabilities = F.softmax(outputs, dim=1)

    # 获取预测类别（取最大概率的索引）
    _, predictions = torch.max(probabilities, dim=1)

    # 确保标签是一维的
    if labels.dim() > 1:
        true_labels = labels.view(-1).long()
    else:
        true_labels = labels.long()

    # 计算混淆矩阵
    # 类别1（霸凌）的指标
    tp = ((predictions == 1) & (true_labels == 1)).sum().item()
    fp = ((predictions == 1) & (true_labels == 0)).sum().item()
    tn = ((predictions == 0) & (true_labels == 0)).sum().item()
    fn = ((predictions == 0) & (true_labels == 1)).sum().item()

    # 计算评估指标
    eps = 1e-7  # 避免除零
    accuracy = (tp + tn) / (tp + tn + fp + fn + eps)

    # 霸凌类的指标
    bullying_precision = tp / (tp + fp + eps)
    bullying_recall = tp / (tp + fn + eps)
    bullying_f1 = 2 * (bullying_precision * bullying_recall) / (bullying_precision + bullying_recall + eps)

    # 非霸凌类的指标
    non_bullying_precision = tn / (tn + fn + eps)
    non_bullying_recall = tn / (tn + fp + eps)
    non_bullying_f1 = 2 * (non_bullying_precision * non_bullying_recall) / (non_bullying_precision + non_bullying_recall + eps)

    # 计算AUC-ROC
    auc_score = 0.5  # 默认值
    if tp + fn > 0 and tn + fp > 0:  # 确保有正负样本
        try:
            from sklearn.metrics import roc_auc_score
            y_true = true_labels.cpu().numpy()
            y_score = probabilities[:, 1].detach().cpu().numpy()  # 使用霸凌类的概率
            auc_score = roc_auc_score(y_true, y_score)
        except (ImportError, ValueError) as e:
            print(f"计算AUC-ROC时出错: {e}")

    metrics = {
        'accuracy': accuracy,
        'bullying': {
            'precision': bullying_precision,
            'recall': bullying_recall,
            'f1': bullying_f1
        },
        'non_bullying': {
            'precision': non_bullying_precision,
            'recall': non_bullying_recall,
            'f1': non_bullying_f1
        },
        'auc': auc_score,
        'confusion_matrix': {
            'true_positive': tp,
            'false_positive': fp,
            'true_negative': tn,
            'false_negative': fn
        }
    }

    return metrics
