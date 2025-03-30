import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, HeteroConv, SAGEConv
from torch_geometric.data import HeteroData
from typing import Dict, List, Tuple
from data_loader import DataLoader
import random

class HeteroGAT(nn.Module):
    """异构图注意力网络"""
    def __init__(self, hidden_dim: int = 128, num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        # 节点特征维度
        self.feature_dims = {
            'user': 1540,        # 768(bert_username) + 768(bert_description) + 4(other_features)
            'media_session': 777,  # 768(bert_description) + 9(other_features)
            'comment': 769,      # 768(bert_text) + 1(postId)
        }
        
        # 边的类型
        self.edge_types = [
            ('user', 'publishes', 'media_session'),
            ('user', 'creates', 'comment'),
            ('comment', 'mentions', 'user'),
            ('comment', 'belongs_to', 'media_session')
        ]
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        # 特征转换层
        self.encoders = nn.ModuleDict({
            node_type: nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            for node_type, dim in self.feature_dims.items()
        })
        
        # 多层GAT
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        for i in range(num_layers):
            # 每一层使用不同类型的卷积以增加多样性
            if i % 2 == 0:
                conv = HeteroConv({
                    edge_type: GATConv(hidden_dim, hidden_dim, heads=4, dropout=dropout, add_self_loops=False)
                    for edge_type in self.edge_types
                })
            else:
                conv = HeteroConv({
                    edge_type: SAGEConv(hidden_dim, hidden_dim)
                    for edge_type in self.edge_types
                })
            
            self.convs.append(conv)
            
            # 为每种节点类型添加批归一化
            batch_norm_dict = nn.ModuleDict({
                node_type: nn.LayerNorm(hidden_dim)
                for node_type in self.feature_dims.keys()
            })
            self.batch_norms.append(batch_norm_dict)
        
        # 注意力机制 - 用于聚合不同节点的信息
        self.attention = nn.ModuleDict({
            node_type: nn.Sequential(
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid()
            )
            for node_type in self.feature_dims.keys()
        })
        
        # 预测层 - 针对媒体会话节点
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2)  # 二分类：霸凌和攻击性
        )
        
        # 初始化分类器权重
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                # 使用更保守的初始化
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                # 最后一层特殊处理
                if m.out_features == 2:
                    nn.init.xavier_normal_(m.weight, gain=0.1)
                    nn.init.constant_(m.bias, -1.0)  # 初始化为负值，让初始预测偏向负类
                else:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
        
        # 打印分类器初始化信息
        with torch.no_grad():
            for name, param in self.classifier.named_parameters():
                if 'weight' in name:
                    print(f"\n分类器层 {name} 初始化范围:")
                    print(f"Min: {param.min().item():.3f}, Max: {param.max().item():.3f}")
                    print(f"Mean: {param.mean().item():.3f}, Std: {param.std().item():.3f}")
        
    def forward(self, x_dict, edge_index_dict):
        """前向传播"""
        # 只在第一次调用时打印图结构信息
        if not hasattr(self, '_printed_structure'):
            print("\n图结构信息:")
            print(f"节点类型: {list(x_dict.keys())}")
            print(f"边类型: {list(edge_index_dict.keys())}")
            self._printed_structure = True
        
        # 检查是否有空的节点特征
        empty_node_types = []
        
        # 处理空节点
        for node_type, x in x_dict.items():
            if x.shape[0] == 0:
                # 为空节点创建一个占位符特征，维度与预期相同
                feature_dim = self.feature_dims.get(node_type, 64)
                x_dict[node_type] = torch.zeros((1, feature_dim), device=x.device)
                empty_node_types.append(node_type)
                
                # 更新相关的边索引，移除涉及此节点的边
                for edge_key in list(edge_index_dict.keys()):
                    src, rel, dst = edge_key
                    if src == node_type or dst == node_type:
                        edge_index_dict[edge_key] = torch.zeros((2, 0), dtype=torch.long, device=x.device)
        
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
        
        # 2. 多层消息传递
        # 创建一个新的边索引字典，排除包含空节点的边
        valid_edge_index_dict = {}
        for edge_type, edge_index in edge_index_dict.items():
            src_type, _, dst_type = edge_type
            if src_type not in empty_node_types and dst_type not in empty_node_types and edge_index.shape[1] > 0:
                valid_edge_index_dict[edge_type] = edge_index
        
        # 如果有有效的边，则进行消息传递
        if valid_edge_index_dict:
            # 只对有效的节点类型进行消息传递
            valid_h_dict = {k: v for k, v in h_dict.items() if k not in empty_node_types}
            
            # 确保所有节点特征都是2维的
            for node_type, feat in valid_h_dict.items():
                if feat.dim() != 2:
                    print(f"Warning: Reshaping {node_type} features from {feat.shape} to 2D")
                    if feat.dim() == 1:
                        valid_h_dict[node_type] = feat.unsqueeze(0)
                    elif feat.dim() > 2:
                        valid_h_dict[node_type] = feat.view(feat.size(0), -1)
            
            # 保存初始特征用于残差连接
            initial_h_dict = {k: v for k, v in valid_h_dict.items()}
            
            # 多层消息传递
            for i in range(self.num_layers):
                # 执行消息传递
                updated_h_dict = self.convs[i](valid_h_dict, valid_edge_index_dict)
                
                # 应用非线性、归一化和残差连接
                for node_type, h in updated_h_dict.items():
                    # 如果是GATConv层，需要正确处理多头注意力的输出
                    if i % 2 == 0:  # GATConv层
                        # 正确的多头注意力处理方式：
                        # 1. 首先reshape为[num_nodes, num_heads, out_channels]
                        h = h.view(-1, 4, self.hidden_dim)  # 4是heads数量
                        # 2. 对注意力头维度取平均
                        h = h.mean(dim=1)  # 现在形状为[num_nodes, hidden_dim]
                    
                    # 应用归一化
                    if h.shape[0] > 1:  # 只有当有多个节点时才应用LayerNorm
                        h = self.batch_norms[i][node_type](h)
                    
                    # 残差连接
                    if node_type in valid_h_dict:
                        h = h + valid_h_dict[node_type]  # 残差连接
                    
                    # 应用ReLU和Dropout
                    h = F.relu(h)
                    h = F.dropout(h, p=self.dropout, training=self.training)
                    
                    # 更新特征
                    valid_h_dict[node_type] = h
            
            # 更新h_dict
            h_dict.update(valid_h_dict)
        
        # 3. 预测 - 使用媒体会话节点的特征来预测标签
        if 'media_session' in h_dict:
            media_features = h_dict['media_session']
            
            # 添加L2正则化
            media_features = F.normalize(media_features, p=2, dim=1)
            
            # 使用注意力权重
            attention_weights = self.attention['media_session'](media_features)
            media_features = media_features * attention_weights
            
            # 跟踪分类器每一层的输出
            x = media_features
            for i, layer in enumerate(self.classifier):
                x = layer(x)
                if isinstance(layer, (nn.Linear, nn.ReLU)) and not hasattr(self, '_first_forward'):
                    print(f"\n分类器_{type(layer).__name__}_{i} 输出范围:")
                    print(f"[{x.min().item():.3f}, {x.max().item():.3f}]")
            
            if not hasattr(self, '_first_forward'):
                self._first_forward = True
            
            return x
        else:
            # 如果没有媒体会话节点，返回零张量
            return torch.zeros((1, 2), device=next(self.parameters()).device)

def calculate_metrics(outputs, labels):
    """计算分类指标"""
    # 添加调试信息
    print("\n预测值统计:")
    print(f"原始输出范围: [{outputs.min().item():.3f}, {outputs.max().item():.3f}]")
    
    probabilities = torch.sigmoid(outputs)
    print(f"Sigmoid后范围: [{probabilities.min().item():.3f}, {probabilities.max().item():.3f}]")
    
    # 动态阈值：使用验证集的预测概率分布来确定
    thresholds = []
    for i in range(probabilities.shape[1]):
        pos_probs = probabilities[labels[:, i] == 1, i]
        neg_probs = probabilities[labels[:, i] == 0, i]
        if len(pos_probs) > 0 and len(neg_probs) > 0:
            threshold = (pos_probs.mean() + neg_probs.mean()) / 2
        else:
            threshold = 0.5
        thresholds.append(threshold)
    
    predictions = torch.zeros_like(probabilities)
    for i in range(probabilities.shape[1]):
        predictions[:, i] = (probabilities[:, i] > thresholds[i]).float()
    
    print(f"使用的阈值: {thresholds}")
    print(f"预测标签分布:\n{predictions.sum(dim=0).cpu().numpy()} / {len(predictions)}")
    print(f"真实标签分布:\n{labels.sum(dim=0).cpu().numpy()} / {len(labels)}")
    
    metrics = {}
    for i, task in enumerate(['bullying', 'aggression']):
        tp = ((predictions[:, i] == 1) & (labels[:, i] == 1)).sum().item()
        fp = ((predictions[:, i] == 1) & (labels[:, i] == 0)).sum().item()
        tn = ((predictions[:, i] == 0) & (labels[:, i] == 0)).sum().item()
        fn = ((predictions[:, i] == 0) & (labels[:, i] == 1)).sum().item()
        
        print(f"\n{task.capitalize()} 混淆矩阵:")
        print(f"True Positive: {tp}, False Positive: {fp}")
        print(f"True Negative: {tn}, False Negative: {fn}")
        
        eps = 1e-7  # 避免除零
        accuracy = (tp + tn) / (tp + tn + fp + fn + eps)
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2 * (precision * recall) / (precision + recall + eps)
        
        # 计算AUC-ROC
        auc_score = 0.5  # 默认值
        if tp + fn > 0 and tn + fp > 0:  # 确保有正负样本
            try:
                from sklearn.metrics import roc_auc_score
                y_true = labels[:, i].cpu().numpy()
                y_score = probabilities[:, i].detach().cpu().numpy()
                auc_score = roc_auc_score(y_true, y_score)
            except (ImportError, ValueError) as e:
                print(f"计算AUC-ROC时出错: {e}")
        
        metrics[task] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc_score,
            'confusion_matrix': {
                'true_positive': tp,
                'false_positive': fp,
                'true_negative': tn,
                'false_negative': fn
            }
        }
    
    return metrics
