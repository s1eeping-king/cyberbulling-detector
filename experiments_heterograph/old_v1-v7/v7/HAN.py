import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HANConv
from torch_geometric.data import HeteroData
from typing import Dict, List, Tuple

class SimpleHAN(nn.Module):
    """异构图注意力网络 (HAN) - 专门处理异构图结构"""
    def __init__(self, hidden_dim: int = 16, num_layers: int = 2, dropout: float = 0.3, heads: int = 4):
        super().__init__()
        # 节点特征维度
        self.feature_dims = {
            'user': 5,           # 5(description_offensive, followerCount_normalized, followingCount_normalized, likeCount_normalized, postCount_normalized)
            'media_session': 74, # 10(原始特征) + 64(BERT特征)
            'comment': 66,       # 2(原始特征: offensive, confidence) + 64(BERT特征)
        }

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

        # 不使用边属性

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.heads = heads

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

        # 多层HAN
        self.han_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()

        # 为每种关系类型定义元路径
        self.metadata = (
            list(self.feature_dims.keys()),  # 节点类型
            [(src, rel, dst) for src, rel, dst in self.relations]  # 边类型
        )

        for i in range(num_layers):
            # 使用HANConv层 - 专门为异构图设计
            self.han_layers.append(
                HANConv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    heads=heads,
                    dropout=dropout,
                    metadata=self.metadata
                )
            )
            # 添加层归一化
            self.layer_norms.append(nn.LayerNorm(hidden_dim))

        # 增强的分类器 - 多层结构
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
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

        # 将最后一层的偏置初始化为更强烈地偏向霸凌类别
        # 这有助于解决类别不平衡问题
        self.classifier[-1].bias.data = torch.tensor([-0.5, 0.5])



    def forward(self, x_dict, edge_index_dict, edge_attr_dict=None):
        """前向传播（使用异构图注意力网络进行消息传递）
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

        # 2. 多层HAN消息传递
        # 过滤edge_index_dict，只保留符合元路径格式的键
        filtered_edge_index_dict = {}
        for edge_key, edge_index in edge_index_dict.items():
            # 跳过非边的键，如'batch_dict'
            if isinstance(edge_key, tuple) and len(edge_key) == 3:
                src_type, edge_type, dst_type = edge_key
                # 确保源节点类型和目标节点类型在h_dict中
                if src_type in h_dict and dst_type in h_dict:
                    filtered_edge_index_dict[edge_key] = edge_index

        # HAN直接处理异构图，不需要将节点合并
        for i in range(self.num_layers):
            # 执行HAN消息传递
            h_dict_new = self.han_layers[i](h_dict, filtered_edge_index_dict)

            # 应用层归一化、ReLU和Dropout
            for node_type, h in h_dict_new.items():
                # 应用层归一化
                h = self.layer_norms[i](h)

                # 应用ReLU
                h = F.relu(h)

                # 应用dropout
                h = F.dropout(h, p=self.dropout, training=self.training)

                # 残差连接（从第二层开始）
                if i > 0:
                    h = h + h_dict[node_type]  # 直接添加残差连接

                h_dict_new[node_type] = h

            # 更新节点特征
            h_dict = h_dict_new

        # 3. 直接使用媒体会话节点作为图的最终表示
        # 这些节点已经通过HAN层接收了来自其他节点的信息
        media_session_features = h_dict['media_session']

        # 应用分类器
        logits = self.classifier(media_session_features)

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
