import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HGTConv, Linear
from torch_geometric.data import HeteroData
from typing import Dict, List, Tuple, Optional

class SimpleHGT(nn.Module):
    """异构图Transformer网络 (HGT) - 使用增强的图结构（评论节点转换为评论边）"""
    def __init__(self, hidden_dim: int = 16, num_layers: int = 1, dropout: float = 0.5, num_heads: int = 4):
        super().__init__()
        # 节点特征维度
        self.feature_dims = {
            'user': 5,           # 5(description_offensive, followerCount_normalized, followingCount_normalized, likeCount_normalized, postCount_normalized)
            'media_session': 74, # 10(原始特征) + 64(BERT特征)
            'comment': 66,       # 2(原始特征: offensive, confidence) + 64(BERT特征)
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

        # 创建元数据
        # HGT需要的元数据格式为 (node_types, edge_types)
        # 其中edge_types是一个元组列表，每个元组为 (src_type, edge_type, dst_type)
        self.metadata = (self.node_types, [(src, edge, dst) for src, edge, dst in self.relations])

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.num_heads = num_heads  # HGT特有参数：注意力头数量

        # 编码器层 - 使用增强的多层感知机将各节点特征编码到hidden_dim维
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

        # HGT层
        self.hgt_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()

        # 创建HGT层
        for _ in range(num_layers):
            self.hgt_layers.append(
                HGTConv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    metadata=self.metadata,
                    heads=num_heads,
                )
            )
            # 添加层归一化
            self.layer_norms.append(nn.LayerNorm(hidden_dim))

        # 简化的分类器 - 减少层数
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2)  # 二分类：[非霸凌, 霸凌]
        )

        # 初始化分类器权重和偏置
        # 使用Xavier初始化权重
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

    def forward(self, x_dict, edge_index_dict, edge_attr_dict=None):
        """前向传播（使用异构图Transformer进行消息传递）
        注意：edge_attr_dict参数保留是为了兼容性，但在此实现中不使用"""
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
                print(f"警告: 未找到节点类型 {node_type} 的编码器")

        # 2. 准备HGT的输入
        # HGT直接使用异构图结构，不需要合并节点

        # 检查是否有节点
        if not h_dict:
            print("警告: 没有找到有效的节点")
            # 返回一个默认的输出
            return torch.zeros((1, 2), device=next(iter(x_dict.values())).device)

        # 3. 多层HGT消息传递
        for i in range(self.num_layers):
            # 创建新的特征字典，用于存储更新后的特征
            h_dict_new = {}

            # 这部分代码不再需要，因为HGTConv会自动处理所有节点类型

            # 收集所有边索引
            edge_dict = {}
            for src_type, edge_type, dst_type in self.relations:
                edge_key = (src_type, edge_type, dst_type)
                if edge_key in edge_index_dict:
                    # HGTConv期望的边索引格式为 (src_type, edge_type, dst_type): edge_index
                    edge_dict[(src_type, edge_type, dst_type)] = edge_index_dict[edge_key]

            # 如果有边，应用HGT层
            if edge_dict:
                # HGT层需要节点特征字典和边索引字典
                # 一次性更新所有节点类型的特征
                h_dict_new = self.hgt_layers[i](h_dict, edge_dict)

                # 对每个节点类型应用层归一化、ReLU和dropout
                for node_type in h_dict_new.keys():
                    if h_dict_new[node_type].size(0) > 0:  # 确保有节点
                        # 应用层归一化
                        h_dict_new[node_type] = self.layer_norms[i](h_dict_new[node_type])

                        # 应用ReLU
                        h_dict_new[node_type] = F.relu(h_dict_new[node_type])

                        # 应用dropout
                        h_dict_new[node_type] = F.dropout(h_dict_new[node_type], p=self.dropout, training=self.training)
            else:
                # 如果没有边，保持原特征
                h_dict_new = h_dict.copy()

            # 更新节点特征
            h_dict = h_dict_new

        # 4. 对每个子图内的评论节点分别进行平均池化
        # 这些评论节点已经通过HGT层接收了来自其他节点的信息
        if 'comment' in h_dict and h_dict['comment'].size(0) > 0 and 'batch_dict' in edge_index_dict and 'comment' in edge_index_dict['batch_dict']:
            # 获取评论节点特征和批次索引
            comment_features = h_dict['comment']
            batch_indices = edge_index_dict['batch_dict']['comment']

            # 获取批次大小（子图数量）
            batch_size = batch_indices.max().item() + 1

            # 创建每个子图的表征
            graph_representations = []

            # 对每个子图分别进行平均池化
            for batch_idx in range(batch_size):
                # 获取当前子图的评论节点索引
                mask = (batch_indices == batch_idx)

                if mask.sum() > 0:  # 如果当前子图有评论节点
                    # 提取当前子图的评论节点特征
                    batch_comment_features = comment_features[mask]

                    # 对当前子图的评论节点进行平均池化
                    batch_representation = batch_comment_features.mean(dim=0)

                    graph_representations.append(batch_representation)
                else:
                    # 如果当前子图没有评论节点，使用零向量
                    graph_representations.append(torch.zeros_like(comment_features[0]))

            # 将所有子图的表征堆叠成一个批次
            graph_representation = torch.stack(graph_representations)

            print(f"使用每个子图的评论节点平均池化作为图表征，批次大小: {batch_size}")
        else:
            # 如果没有评论节点或批次信息，回退到使用所有节点的平均特征
            print("警告: 没有评论节点或批次信息，使用所有节点的平均特征")

            # 检查是否有批次信息
            if 'batch_dict' in edge_index_dict:
                # 获取批次大小
                batch_size = 0
                for node_type in edge_index_dict['batch_dict']:
                    if edge_index_dict['batch_dict'][node_type].size(0) > 0:
                        batch_size = max(batch_size, edge_index_dict['batch_dict'][node_type].max().item() + 1)

                # 创建每个子图的表征
                graph_representations = []

                # 对每个子图分别处理
                for batch_idx in range(batch_size):
                    batch_features = []

                    # 收集当前子图的所有节点特征
                    for node_type in h_dict:
                        if node_type in edge_index_dict['batch_dict']:
                            mask = (edge_index_dict['batch_dict'][node_type] == batch_idx)
                            if mask.sum() > 0:
                                batch_features.append(h_dict[node_type][mask])

                    if batch_features:
                        # 合并当前子图的所有节点特征并计算平均值
                        all_batch_features = torch.cat(batch_features, dim=0)
                        batch_representation = all_batch_features.mean(dim=0)
                    else:
                        # 如果当前子图没有节点，使用零向量
                        feature_dim = next(iter(h_dict.values())).size(1)
                        batch_representation = torch.zeros(feature_dim, device=next(iter(h_dict.values())).device)

                    graph_representations.append(batch_representation)

                # 将所有子图的表征堆叠成一个批次
                graph_representation = torch.stack(graph_representations)
            else:
                # 如果没有批次信息，只能使用所有节点的平均特征
                all_features = torch.cat([h for h in h_dict.values()], dim=0)
                # 创建一个单元素批次
                graph_representation = all_features.mean(dim=0).unsqueeze(0)

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
