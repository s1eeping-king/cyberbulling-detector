import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from typing import Dict, List, Tuple, Optional

class SimpleGAT(nn.Module):
    """异构图注意力网络 (GAT) - 使用多节点类型融合的图表征

    架构特点：
    1. 对每个子图内的评论、用户、媒体会话节点分别进行平均池化
    2. 将三种节点类型的池化特征拼接
    3. 通过图融合层降维到hidden_dim
    4. 送入分类器进行霸凌检测
    """
    def __init__(self, hidden_dim: int = 16, num_layers: int = 1, dropout: float = 0.5, heads: int = 4):
        super().__init__()
        # 节点特征维度 - 确保与data_loader.py中的定义一致
        self.feature_dims = {
            'user': 34,           # 5(基本特征) + 29(扩展特征)
            'media_session': 778, # 10(原始特征) + 768(RoBERTa特征)
            'comment': 770,       # 2(原始特征: offensive, confidence) + 768(RoBERTa特征)
        }

        # 节点类型
        self.node_types = list(self.feature_dims.keys())

        # 关系类型 - 混合图结构（同时包含评论节点和用户间直接边）
        self.relations = [
            # 原始关系
            ('user', 'publishes', 'media_session'),
            ('user', 'creates', 'comment'),
            ('comment', 'belongs_to', 'media_session'),
            ('comment', 'mentions', 'user'),
            # 新增的用户间直接边
            ('user', 'offensive_comment', 'user'),
            ('user', 'non_offensive_comment', 'user')
        ]

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.heads = heads

        # 编码器层 - 将各节点特征编码到hidden_dim维
        self.encoders = nn.ModuleDict({
            node_type: nn.Sequential(
                nn.Linear(dim, hidden_dim * 2),  # 先扩展到更大的维度
                nn.LayerNorm(hidden_dim * 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, hidden_dim),  # 然后映射到目标维度
                nn.LayerNorm(hidden_dim),
                nn.ReLU()
            )
            for node_type, dim in self.feature_dims.items()
        })

        # 创建统一的GAT层
        self.gat_layers = nn.ModuleList()
        for i in range(num_layers):
            in_channels = hidden_dim if i == 0 else hidden_dim * heads
            self.gat_layers.append(
                GATConv(
                    in_channels=in_channels,
                    out_channels=hidden_dim,
                    heads=heads,
                    dropout=dropout,
                    concat=True if i < num_layers - 1 else False
                )
            )

        # 图表征融合层 - 将评论、用户、媒体会话的池化特征拼接后降维
        # 拼接后的维度为 hidden_dim * 3 (评论 + 用户 + 媒体会话)
        self.graph_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )

        # 简化的分类器
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
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

        # 使用中立的初始化，不偏向任何类别
        self.classifier[-1].bias.data = torch.zeros(2)

        # 对最后一层的权重进行特殊初始化，使其对两个类别的预测更加平衡
        # 这有助于防止模型在训练初期就偏向某一类别
        last_layer = self.classifier[-1]
        with torch.no_grad():
            # 对权重进行归一化，使得每个输出神经元的权重和相近
            weight_norm = last_layer.weight.norm(dim=1, keepdim=True)
            last_layer.weight.div_(weight_norm + 1e-6)

    def forward(self, x_dict, edge_index_dict, edge_attr_dict=None):
        """前向传播（使用GAT进行消息传递）"""
        # 检查是否有空的节点特征
        for node_type, x in x_dict.items():
            if x.shape[0] == 0:
                feature_dim = self.feature_dims.get(node_type, self.hidden_dim)
                x_dict[node_type] = torch.zeros((1, feature_dim), device=x.device)

        # 1. 特征编码 - 将各节点特征编码到hidden_dim维
        h_dict = {}
        for node_type, x in x_dict.items():
            if node_type in self.encoders:
                h_dict[node_type] = self.encoders[node_type](x)  # 编码器已经包含ReLU激活
            else:
                # 如果没有找到编码器，使用默认的线性层
                h_dict[node_type] = nn.Linear(x.size(1), self.hidden_dim).to(x.device)(x)
                # 移除不必要的输出
                # print(f"警告: 未找到节点类型 {node_type} 的编码器")

        # 检查是否有节点
        if not h_dict:
            # 移除不必要的输出
            # print("警告: 没有找到有效的节点")
            # 返回一个默认的输出
            return torch.zeros((1, 2), device=next(iter(x_dict.values())).device)

        # 2. 将所有节点特征合并，并收集所有边
        # 合并所有节点特征
        node_features = []
        node_type_indices = {}  # 记录每种节点类型在合并后的特征中的起始索引
        current_idx = 0

        for node_type in self.node_types:
            if node_type in h_dict and h_dict[node_type].size(0) > 0:
                node_type_indices[node_type] = (current_idx, current_idx + h_dict[node_type].size(0))
                node_features.append(h_dict[node_type])
                current_idx += h_dict[node_type].size(0)

        if not node_features:
            # 移除不必要的输出
            # print("警告: 没有有效的节点特征")
            return torch.zeros((1, 2), device=next(iter(x_dict.values())).device)

        # 合并所有节点特征
        x = torch.cat(node_features, dim=0)

        # 收集并重新映射所有边
        edge_indices = []

        for src, edge_type, dst in self.relations:
            edge_key = (src, edge_type, dst)
            if edge_key in edge_index_dict and edge_index_dict[edge_key].size(1) > 0:
                if src in node_type_indices and dst in node_type_indices:
                    # 获取源节点和目标节点在合并特征中的索引范围
                    src_start, _ = node_type_indices[src]
                    dst_start, _ = node_type_indices[dst]

                    # 重新映射边的索引
                    edge_index = edge_index_dict[edge_key].clone()
                    edge_index[0, :] += src_start
                    edge_index[1, :] += dst_start

                    edge_indices.append(edge_index)

        # 如果有边，应用GAT层
        if edge_indices:
            # 合并所有边
            edge_index = torch.cat(edge_indices, dim=1)

            # 应用GAT层
            for i, gat_layer in enumerate(self.gat_layers):
                x = gat_layer(x, edge_index)
                if i < self.num_layers - 1:  # 不在最后一层应用激活和dropout
                    x = F.relu(x)
                    x = F.dropout(x, p=self.dropout, training=self.training)

            # 将处理后的特征分配回各节点类型
            for node_type, (start_idx, end_idx) in node_type_indices.items():
                h_dict[node_type] = x[start_idx:end_idx]

        # 3. 对每个子图内的评论、用户、媒体会话节点分别进行平均池化并拼接
        graph_representation = None  # 初始化为None，确保在所有分支中都会被赋值

        if 'batch_dict' in edge_index_dict:
            # 获取批次大小
            batch_size = 0
            for node_type in edge_index_dict['batch_dict']:
                if edge_index_dict['batch_dict'][node_type].size(0) > 0:
                    batch_size = max(batch_size, edge_index_dict['batch_dict'][node_type].max().item() + 1)

            if batch_size > 0:
                # 创建每个子图的表征
                graph_representations = []

                # 对每个子图分别进行多节点类型的平均池化
                for batch_idx in range(batch_size):
                    # 初始化三种节点类型的池化特征
                    comment_pooled = torch.zeros(self.hidden_dim, device=next(iter(h_dict.values())).device)
                    user_pooled = torch.zeros(self.hidden_dim, device=next(iter(h_dict.values())).device)
                    media_pooled = torch.zeros(self.hidden_dim, device=next(iter(h_dict.values())).device)

                    # 1. 评论节点平均池化
                    if 'comment' in h_dict and h_dict['comment'].size(0) > 0 and 'comment' in edge_index_dict['batch_dict']:
                        comment_batch_indices = edge_index_dict['batch_dict']['comment']
                        comment_mask = (comment_batch_indices == batch_idx)
                        if comment_mask.sum() > 0:
                            batch_comment_features = h_dict['comment'][comment_mask]
                            comment_pooled = batch_comment_features.mean(dim=0)

                    # 2. 用户节点平均池化
                    if 'user' in h_dict and h_dict['user'].size(0) > 0 and 'user' in edge_index_dict['batch_dict']:
                        user_batch_indices = edge_index_dict['batch_dict']['user']
                        user_mask = (user_batch_indices == batch_idx)
                        if user_mask.sum() > 0:
                            batch_user_features = h_dict['user'][user_mask]
                            user_pooled = batch_user_features.mean(dim=0)

                    # 3. 媒体会话节点平均池化
                    if 'media_session' in h_dict and h_dict['media_session'].size(0) > 0 and 'media_session' in edge_index_dict['batch_dict']:
                        media_batch_indices = edge_index_dict['batch_dict']['media_session']
                        media_mask = (media_batch_indices == batch_idx)
                        if media_mask.sum() > 0:
                            batch_media_features = h_dict['media_session'][media_mask]
                            media_pooled = batch_media_features.mean(dim=0)

                    # 4. 拼接三种节点类型的池化特征
                    concatenated_features = torch.cat([comment_pooled, user_pooled, media_pooled], dim=0)

                    # 5. 通过融合层降维
                    fused_representation = self.graph_fusion(concatenated_features)

                    graph_representations.append(fused_representation)

                # 将所有子图的表征堆叠成一个批次
                graph_representation = torch.stack(graph_representations)

                # 移除不必要的输出
                # print(f"使用评论、用户、媒体会话节点的融合池化作为图表征，批次大小: {batch_size}")
            else:
                # 如果batch_size为0，使用默认的回退方案
                # 创建默认的零向量表征
                default_representation = torch.zeros(self.hidden_dim, device=next(iter(h_dict.values())).device)
                graph_representation = default_representation.unsqueeze(0)
        else:
            # 如果没有批次信息，使用所有节点的平均特征作为回退方案
            if h_dict:
                all_features = torch.cat([h for h in h_dict.values()], dim=0)
                # 创建默认的表征（使用所有节点的平均特征）
                avg_features = all_features.mean(dim=0)
                # 由于没有三种节点类型的分离，我们复制三次然后通过融合层
                concatenated_features = torch.cat([avg_features, avg_features, avg_features], dim=0)
                fused_representation = self.graph_fusion(concatenated_features)
                graph_representation = fused_representation.unsqueeze(0)
            else:
                # 如果完全没有节点，创建零向量
                default_representation = torch.zeros(self.hidden_dim, device=next(iter(x_dict.values())).device)
                graph_representation = default_representation.unsqueeze(0)

        # 确保graph_representation不为None
        if graph_representation is None:
            # 如果graph_representation仍然为None，使用默认的零向量
            default_representation = torch.zeros(self.hidden_dim, device=next(iter(x_dict.values())).device)
            graph_representation = default_representation.unsqueeze(0)

        # 应用图表征融合层（如果还没有应用的话）
        # 检查graph_representation的维度，如果不是hidden_dim，说明还没有通过融合层
        if graph_representation.size(-1) != self.hidden_dim:
            # 如果维度不匹配，说明这是一个回退情况，需要通过融合层
            if graph_representation.size(-1) == self.hidden_dim * 3:
                # 如果是拼接后的维度，直接通过融合层
                graph_representation = self.graph_fusion(graph_representation)
            else:
                # 否则，复制三次然后通过融合层
                expanded_repr = graph_representation.repeat(1, 3)
                graph_representation = self.graph_fusion(expanded_repr)

        # 应用分类器
        logits = self.classifier(graph_representation)

        return logits

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
