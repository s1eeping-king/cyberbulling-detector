import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleClassifier(nn.Module):
    """多节点类型聚合分类器 - 分别对评论、用户、媒体会话节点进行聚合，然后拼接进行分类"""
    def __init__(self, hidden_dim: int = 16, dropout: float = 0.5):
        super().__init__()
        # 节点特征维度 - 确保与data_loader.py中的定义一致
        self.feature_dims = {
            'user': 34,           # 5(基本特征) + 29(扩展特征)
            'media_session': 778, # 10(原始特征) + 768(RoBERTa特征)
            'comment': 770,       # 2(原始特征: offensive, confidence) + 768(RoBERTa特征)
        }

        # 节点类型
        self.node_types = list(self.feature_dims.keys())

        self.hidden_dim = hidden_dim
        self.dropout = dropout

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

        # 多节点类型聚合的分类器 - 输入维度为 hidden_dim * 3 (评论 + 用户 + 媒体会话)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),  # 先扩展维度
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),      # 降维
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
        """前向传播（多节点类型聚合）

        流程：
        1. 特征编码：将各节点类型特征编码到统一维度
        2. 分别聚合：对评论、用户、媒体会话节点分别进行平均池化
        3. 特征拼接：将三种聚合特征拼接成最终表征
        4. 分类预测：通过分类器进行霸凌检测
        """
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

        # 检查是否有节点
        if not h_dict:
            # 返回一个默认的输出
            return torch.zeros((1, 2), device=next(iter(x_dict.values())).device)

        # 2. 对每个子图内的三种节点类型分别进行平均池化，然后拼接
        graph_representation = None

        if 'batch_dict' in edge_index_dict:
            # 获取批次大小（子图数量）
            batch_size = 0
            for node_type in edge_index_dict['batch_dict']:
                if edge_index_dict['batch_dict'][node_type].size(0) > 0:
                    batch_size = max(batch_size, edge_index_dict['batch_dict'][node_type].max().item() + 1)

            if batch_size > 0:
                # 创建每个子图的表征
                graph_representations = []

                # 对每个子图分别处理
                for batch_idx in range(batch_size):
                    # 存储三种节点类型的聚合特征
                    node_type_representations = []

                    # 对每种节点类型分别进行聚合
                    for node_type in ['comment', 'user', 'media_session']:
                        if (node_type in h_dict and h_dict[node_type].size(0) > 0 and
                            node_type in edge_index_dict['batch_dict']):
                            # 获取当前子图的该类型节点索引
                            batch_indices = edge_index_dict['batch_dict'][node_type]
                            mask = (batch_indices == batch_idx)

                            if mask.sum() > 0:  # 如果当前子图有该类型节点
                                # 提取当前子图的该类型节点特征
                                batch_node_features = h_dict[node_type][mask]
                                # 对当前子图的该类型节点进行平均池化
                                node_representation = batch_node_features.mean(dim=0)
                            else:
                                # 如果当前子图没有该类型节点，使用零向量
                                node_representation = torch.zeros(self.hidden_dim, device=h_dict[node_type].device)
                        else:
                            # 如果该类型节点不存在，使用零向量
                            device = next(iter(h_dict.values())).device
                            node_representation = torch.zeros(self.hidden_dim, device=device)

                        node_type_representations.append(node_representation)

                    # 拼接三种节点类型的聚合特征
                    batch_representation = torch.cat(node_type_representations, dim=0)
                    graph_representations.append(batch_representation)

                # 将所有子图的表征堆叠成一个批次
                graph_representation = torch.stack(graph_representations)

        # 确保graph_representation不为None
        if graph_representation is None:
            # 如果没有批次信息，创建一个单元素批次，使用三种节点类型的平均特征拼接
            device = next(iter(h_dict.values())).device
            node_type_representations = []

            # 对每种节点类型分别计算平均特征
            for node_type in ['comment', 'user', 'media_session']:
                if node_type in h_dict and h_dict[node_type].size(0) > 0:
                    # 计算该类型节点的平均特征
                    node_representation = h_dict[node_type].mean(dim=0)
                else:
                    # 如果该类型节点不存在，使用零向量
                    node_representation = torch.zeros(self.hidden_dim, device=device)

                node_type_representations.append(node_representation)

            # 拼接三种节点类型的特征
            combined_representation = torch.cat(node_type_representations, dim=0)
            # 创建一个单元素批次
            graph_representation = combined_representation.unsqueeze(0)

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
