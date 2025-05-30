import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import HeteroData
from typing import Dict, List, Tuple

class SimpleGAT(nn.Module):
    """简化版图注意力网络 - 使用普通GAT而不是RGAT"""
    def __init__(self, hidden_dim: int = 16, num_layers: int = 2, dropout: float = 0.3, heads: int = 4):
        super().__init__()
        # 节点特征维度
        self.feature_dims = {
            'user': 5,           # 5(description_offensive, followerCount_normalized, followingCount_normalized, likeCount_normalized, postCount_normalized)
            'media_session': 74, # 10(原始特征) + 64(BERT特征)
            'comment': 66        # 2(原始特征) + 64(BERT特征)
        }

        # 不再需要显式定义关系类型，因为我们使用普通GAT

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

        # 多层GAT - 将所有节点视为同构图，共用GAT层
        self.gat_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()

        for i in range(num_layers):
            # 为所有节点创建一个共同的GAT层
            self.gat_layers.append(
                GATConv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim // heads,
                    heads=heads,
                    dropout=dropout,
                    add_self_loops=True,  # 在同构图中可以添加自环
                    concat=True  # 连接多头注意力的输出
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



    def forward(self, x_dict, edge_index_dict):
        """前向传播（使用普通图注意力网络进行消息传递）"""
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

        # 2. 多层GAT消息传递 - 将所有节点合并为同构图，但保留原始边结构
        # 合并所有节点特征
        all_nodes = []
        node_type_indices = {}  # 记录每种节点类型在合并后的索引范围
        start_idx = 0

        for node_type, h in h_dict.items():
            num_nodes = h.size(0)
            if num_nodes > 0:
                all_nodes.append(h)
                node_type_indices[node_type] = (start_idx, start_idx + num_nodes)
                start_idx += num_nodes

        # 如果有节点
        if all_nodes:
            # 合并所有节点特征
            all_features = torch.cat(all_nodes, dim=0)

            # 使用原始边结构，但不区分边类型
            # 收集所有边并调整索引
            all_edges = []

            # 遍历所有边类型
            for edge_key, edges in edge_index_dict.items():
                # 跳过非边的键
                if edge_key == 'batch_dict' or not isinstance(edges, torch.Tensor):
                    continue

                # 获取源节点类型和目标节点类型
                if isinstance(edge_key, tuple) and len(edge_key) == 3:
                    src_type, _, dst_type = edge_key

                    # 获取源节点和目标节点的偏移量
                    src_offset = node_type_indices.get(src_type, (0, 0))[0]
                    dst_offset = node_type_indices.get(dst_type, (0, 0))[0]

                    # 调整边的索引
                    if edges.size(0) == 2 and edges.size(1) > 0:  # 确保边是有效的
                        src_nodes = edges[0] + src_offset
                        dst_nodes = edges[1] + dst_offset

                        # 添加到所有边列表
                        edge = torch.stack([src_nodes, dst_nodes], dim=0)
                        all_edges.append(edge)

            # 如果没有边，创建自环
            if len(all_edges) == 0:
                print("警告: 没有找到有效的边，创建自环")
                num_total_nodes = all_features.size(0)
                edge_index = torch.arange(num_total_nodes, device=all_features.device)
                edge_index = torch.stack([edge_index, edge_index], dim=0)
            else:
                # 合并所有边
                edge_index = torch.cat(all_edges, dim=1)

                # 添加自环以确保每个节点至少有一条边
                # 注意：GATConv已经有add_self_loops=True，所以这里可以不添加
                # 但为了安全起见，我们仍然添加
                num_total_nodes = all_features.size(0)
                self_loops = torch.arange(num_total_nodes, device=all_features.device)
                self_loops = torch.stack([self_loops, self_loops], dim=0)
                edge_index = torch.cat([edge_index, self_loops], dim=1)

                # 去除重复的边
                edge_index = torch.unique(edge_index, dim=1)

            # 3. 多层GAT消息传递
            h = all_features
            for i in range(self.num_layers):
                # 执行GAT消息传递
                h_new = self.gat_layers[i](h, edge_index)

                # 处理多头注意力输出
                if h_new.size(-1) != self.hidden_dim:
                    # 如果输出维度不匹配，可能是因为多头注意力的输出被连接了
                    # 需要将其转换回原始维度
                    h_new = h_new.view(h_new.size(0), -1)  # 展平多头输出
                    if h_new.size(-1) != self.hidden_dim:
                        # 如果维度仍然不匹配，使用线性投影
                        h_new = nn.Linear(h_new.size(-1), self.hidden_dim, device=h.device)(h_new)

                # 应用层归一化
                h_new = self.layer_norms[i](h_new)

                # 应用ReLU
                h_new = F.relu(h_new)

                # 应用dropout
                h_new = F.dropout(h_new, p=self.dropout, training=self.training)

                # 简单的残差连接
                if i > 0:  # 从第二层开始应用残差连接
                    h_new = h_new + h  # 直接添加残差连接

                h = h_new

            # 将更新后的特征分配回各节点类型
            updated_h_dict = {}
            for node_type, (start, end) in node_type_indices.items():
                updated_h_dict[node_type] = h[start:end]

            # 更新节点特征
            h_dict = updated_h_dict

        # 4. 直接使用媒体会话节点作为图的最终表示
        # 这些节点已经通过GAT层接收了来自其他节点的信息
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
