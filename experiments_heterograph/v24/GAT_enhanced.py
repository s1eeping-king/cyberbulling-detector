import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from typing import Dict, List, Tuple, Optional

class EnhancedGAT(nn.Module):
    """增强版GAT - 添加注意力机制和特征增强"""
    
    def __init__(self, hidden_dim: int = 16, num_layers: int = 1, dropout: float = 0.5, heads: int = 4):
        super().__init__()
        
        # 节点特征维度
        self.feature_dims = {
            'user': 34,
            'media_session': 778,
            'comment': 770,
        }
        
        self.node_types = list(self.feature_dims.keys())
        self.relations = [
            ('user', 'publishes', 'media_session'),
            ('user', 'creates', 'comment'),
            ('comment', 'belongs_to', 'media_session'),
            ('comment', 'mentions', 'user'),
            ('user', 'offensive_comment', 'user'),
            ('user', 'non_offensive_comment', 'user')
        ]
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.heads = heads

        # 增强的编码器 - 添加残差连接
        self.encoders = nn.ModuleDict({
            node_type: nn.Sequential(
                nn.Linear(dim, hidden_dim * 2),
                nn.LayerNorm(hidden_dim * 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU()
            )
            for node_type, dim in self.feature_dims.items()
        })

        # GAT层
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

        # 节点类型特定的注意力权重
        self.node_attention = nn.ModuleDict({
            node_type: nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1),
                nn.Sigmoid()
            )
            for node_type in self.node_types
        })

        # 改进的图融合层 - 添加更多层和残差连接
        self.graph_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 4),
            nn.LayerNorm(hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )

        # 增强的分类器
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2)
        )

        self._init_weights()

    def _init_weights(self):
        """改进的权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        # 分类器最后一层的特殊初始化
        last_layer = self.classifier[-1]
        nn.init.xavier_uniform_(last_layer.weight, gain=0.1)
        nn.init.zeros_(last_layer.bias)

    def forward(self, x_dict, edge_index_dict, edge_attr_dict=None):
        """增强的前向传播"""
        
        # 检查空节点特征
        for node_type, x in x_dict.items():
            if x.shape[0] == 0:
                feature_dim = self.feature_dims.get(node_type, self.hidden_dim)
                x_dict[node_type] = torch.zeros((1, feature_dim), device=x.device)

        # 1. 特征编码
        h_dict = {}
        for node_type, x in x_dict.items():
            if node_type in self.encoders:
                h_dict[node_type] = self.encoders[node_type](x)
            else:
                h_dict[node_type] = nn.Linear(x.size(1), self.hidden_dim).to(x.device)(x)

        if not h_dict:
            return torch.zeros((1, 2), device=next(iter(x_dict.values())).device)

        # 2. GAT消息传递
        node_features = []
        node_type_indices = {}
        current_idx = 0

        for node_type in self.node_types:
            if node_type in h_dict and h_dict[node_type].size(0) > 0:
                node_type_indices[node_type] = (current_idx, current_idx + h_dict[node_type].size(0))
                node_features.append(h_dict[node_type])
                current_idx += h_dict[node_type].size(0)

        if not node_features:
            return torch.zeros((1, 2), device=next(iter(x_dict.values())).device)

        x = torch.cat(node_features, dim=0)

        # 收集边
        edge_indices = []
        for src, edge_type, dst in self.relations:
            edge_key = (src, edge_type, dst)
            if edge_key in edge_index_dict and edge_index_dict[edge_key].size(1) > 0:
                if src in node_type_indices and dst in node_type_indices:
                    src_start, _ = node_type_indices[src]
                    dst_start, _ = node_type_indices[dst]
                    
                    edge_index = edge_index_dict[edge_key].clone()
                    edge_index[0, :] += src_start
                    edge_index[1, :] += dst_start
                    
                    edge_indices.append(edge_index)

        # 应用GAT层
        if edge_indices:
            edge_index = torch.cat(edge_indices, dim=1)
            
            for i, gat_layer in enumerate(self.gat_layers):
                x_new = gat_layer(x, edge_index)
                if i < self.num_layers - 1:
                    x_new = F.relu(x_new)
                    x_new = F.dropout(x_new, p=self.dropout, training=self.training)
                x = x_new

            # 将处理后的特征分配回各节点类型
            for node_type, (start_idx, end_idx) in node_type_indices.items():
                h_dict[node_type] = x[start_idx:end_idx]

        # 3. 增强的图表征生成
        graph_representation = None

        if 'batch_dict' in edge_index_dict:
            batch_size = 0
            for node_type in edge_index_dict['batch_dict']:
                if edge_index_dict['batch_dict'][node_type].size(0) > 0:
                    batch_size = max(batch_size, edge_index_dict['batch_dict'][node_type].max().item() + 1)

            if batch_size > 0:
                graph_representations = []

                for batch_idx in range(batch_size):
                    # 初始化节点池化特征
                    pooled_features = {}
                    
                    for node_type in ['comment', 'user', 'media_session']:
                        if (node_type in h_dict and h_dict[node_type].size(0) > 0 and 
                            node_type in edge_index_dict['batch_dict']):
                            
                            batch_indices = edge_index_dict['batch_dict'][node_type]
                            mask = (batch_indices == batch_idx)
                            
                            if mask.sum() > 0:
                                batch_features = h_dict[node_type][mask]
                                
                                # 应用注意力权重
                                attention_weights = self.node_attention[node_type](batch_features)
                                weighted_features = batch_features * attention_weights
                                pooled = weighted_features.mean(dim=0)
                            else:
                                pooled = torch.zeros(self.hidden_dim, device=next(iter(h_dict.values())).device)
                        else:
                            pooled = torch.zeros(self.hidden_dim, device=next(iter(h_dict.values())).device)
                        
                        pooled_features[node_type] = pooled

                    # 拼接特征
                    concatenated = torch.cat([
                        pooled_features['comment'],
                        pooled_features['user'], 
                        pooled_features['media_session']
                    ], dim=0)
                    
                    # 通过融合层
                    fused = self.graph_fusion(concatenated)
                    graph_representations.append(fused)

                graph_representation = torch.stack(graph_representations)
            else:
                default_repr = torch.zeros(self.hidden_dim, device=next(iter(h_dict.values())).device)
                graph_representation = default_repr.unsqueeze(0)
        else:
            # 回退方案
            if h_dict:
                all_features = torch.cat([h for h in h_dict.values()], dim=0)
                avg_features = all_features.mean(dim=0)
                concatenated = torch.cat([avg_features, avg_features, avg_features], dim=0)
                fused = self.graph_fusion(concatenated)
                graph_representation = fused.unsqueeze(0)
            else:
                default_repr = torch.zeros(self.hidden_dim, device=next(iter(x_dict.values())).device)
                graph_representation = default_repr.unsqueeze(0)

        # 确保不为None
        if graph_representation is None:
            default_repr = torch.zeros(self.hidden_dim, device=next(iter(x_dict.values())).device)
            graph_representation = default_repr.unsqueeze(0)

        # 应用分类器
        logits = self.classifier(graph_representation)
        return logits
