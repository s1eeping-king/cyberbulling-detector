import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HGTConv, Linear
from typing import Dict, List, Optional
import numpy as np

class KGEmbeddingLayer(nn.Module):
    def __init__(self, num_entities, embedding_dim):
        super().__init__()
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_entities, embedding_dim)
        
    def forward(self, entities, relations=None):
        entity_emb = self.entity_embeddings(entities)
        if relations is not None:
            relation_emb = self.relation_embeddings(relations)
            return entity_emb, relation_emb
        return entity_emb

class FeatureFusion(nn.Module):
    def __init__(self, feature_dim, kg_dim):
        super().__init__()
        self.attention = nn.Linear(feature_dim + kg_dim, 1)
        self.feature_proj = nn.Linear(feature_dim, feature_dim)
        self.kg_proj = nn.Linear(kg_dim, feature_dim)
        
    def forward(self, features, kg_embeddings):
        # 投影特征
        proj_features = self.feature_proj(features)
        proj_kg = self.kg_proj(kg_embeddings)
        
        # 计算注意力权重
        combined = torch.cat([features, kg_embeddings], dim=-1)
        attention_weights = torch.sigmoid(self.attention(combined))
        
        # 融合特征
        fused_features = attention_weights * proj_features + (1 - attention_weights) * proj_kg
        return fused_features

class HGTransformer(nn.Module):
    """异构图Transformer网络 - 使用super节点的单向消息传递"""
    def __init__(self, hidden_dim: int = 128, num_layers: int = 2, num_heads: int = 4, dropout: float = 0.3, num_node_types=4):
        super().__init__()
        # 节点特征维度
        self.feature_dims = {
            'user': 260,         # 128(bert_username) + 128(bert_description) + 4(other_features)
            'media_session': 137, # 128(bert_description) + 9(other_features)
            'comment': 129,      # 128(bert_text) + 1(postId)
            'super': hidden_dim  # super节点初始维度与hidden_dim相同
        }
        
        # 节点类型和元关系定义
        self.node_types = ['user', 'media_session', 'comment', 'super']
        self.edge_types = [
            # 基础消息传递边
            ('user', 'publishes', 'media_session'),
            ('user', 'creates', 'comment'),
            ('comment', 'belongs_to', 'media_session'),
            ('comment', 'mentions', 'user'),
            # 到super节点的边
            ('user', 'to_super', 'super'),
            ('media_session', 'to_super', 'super'),
            ('comment', 'to_super', 'super')
        ]
        
        self.metadata = (self.node_types, self.edge_types)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        
        # 知识图谱嵌入层
        self.kg_embedding = KGEmbeddingLayer(num_entities=10000, embedding_dim=64)  # 减小KG嵌入维度
        
        # 特征融合层
        self.fusion_layers = nn.ModuleDict({
            'user': FeatureFusion(260, 64),        # 用户节点
            'media_session': FeatureFusion(137, 64),  # 媒体会话节点
            'comment': FeatureFusion(129, 64),     # 评论节点
            'super': FeatureFusion(128, 64)        # 超级节点
        })
        
        # 节点类型编码器
        self.node_encoders = nn.ModuleDict({
            'user': nn.Sequential(
                Linear(260, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ),
            'media_session': nn.Sequential(
                Linear(137, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ),
            'comment': nn.Sequential(
                Linear(129, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ),
            'super': nn.Sequential(
                Linear(128, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
        })
        
        # HGT层
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            conv = HGTConv(
                in_channels={node_type: hidden_dim for node_type in self.node_types},
                out_channels=hidden_dim,
                metadata=self.metadata,
                heads=num_heads
            )
            self.layers.append(conv)
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2)
        )
        
        # 移除attention_stats字典，改用layer_outputs来存储中间输出
        self.feature_stats = {}
        self.layer_outputs = {}
        
        # 添加注意力权重存储
        self.attention_weights = {}
        
    def monitor_feature_statistics(self, h_dict: Dict[str, torch.Tensor]) -> Dict:
        """监控各类型节点特征的统计信息"""
        stats = {}
        for node_type, features in h_dict.items():
            if features.numel() == 0:
                continue
            stats[node_type] = {
                'mean': features.mean().item(),
                'std': features.std().item(),
                'max': features.max().item(),
                'min': features.min().item(),
                'norm': features.norm().item(),
                'active_ratio': (features > 0).float().mean().item()  # ReLU激活率
            }
        return stats
        
    def get_layer_statistics(self, h_dict_new: Dict[str, torch.Tensor], layer_idx: int) -> Dict:
        """获取每一层的输出统计信息"""
        layer_stats = {}
        for node_type, features in h_dict_new.items():
            if features.numel() == 0:
                continue
            
            # 计算基本统计量
            layer_stats[node_type] = {
                'mean': features.mean().item(),
                'std': features.std().item(),
                'max': features.max().item(),
                'min': features.min().item(),
                'norm': features.norm().item(),
                'active_ratio': (features > 0).float().mean().item(),
                'grad_norm': 0.0  # 在backward时更新
            }
            
            # 如果在训练模式下，注册钩子来收集梯度信息
            if self.training and features.requires_grad:
                def hook_fn(grad, stats=layer_stats[node_type]):
                    stats['grad_norm'] = grad.norm().item()
                features.register_hook(hook_fn)
        
        return layer_stats
        
    def forward(self, x_dict: Dict[str, torch.Tensor], 
               edge_index_dict: Dict[str, torch.Tensor],
               node_type_ids=None) -> torch.Tensor:
        """前向传播"""
        # 获取知识图谱嵌入
        if node_type_ids is not None:
            kg_embeddings = {
                node_type: self.kg_embedding(ids) 
                for node_type, ids in node_type_ids.items()
            }
        else:
            # 如果没有提供节点ID，使用零向量
            kg_embeddings = {
                node_type: torch.zeros((x.shape[0], 64), device=x.device)
                for node_type, x in x_dict.items()
            }
        
        # 特征融合
        h_dict = {}
        batch_size = None
        for node_type, x in x_dict.items():
            if node_type == 'super':
                batch_size = x.size(0)
                h_dict[node_type] = self.node_encoders[node_type](x)
                continue
                
            if x.shape[0] == 0:
                h_dict[node_type] = torch.zeros((1, self.hidden_dim), device=x.device)
            else:
                # 融合原始特征和知识图谱嵌入
                fused_features = self.fusion_layers[node_type](x, kg_embeddings[node_type])
                # 编码到隐藏维度
                h_dict[node_type] = self.node_encoders[node_type](fused_features)
        
        # 记录初始特征统计
        self.feature_stats['initial'] = self.monitor_feature_statistics(h_dict)
        
        # 多层HGT消息传递
        for i in range(self.num_layers):
            h_dict_new = {}
            
            # 在消息传递前应用dropout
            h_dict_dropped = {
                k: F.dropout(v, p=self.dropout, training=self.training)
                for k, v in h_dict.items()
            }
            
            # HGT消息传递
            out_dict = self.layers[i](h_dict_dropped, edge_index_dict)
            
            # 存储当前层的注意力权重
            if hasattr(self.layers[i], '_alpha'):
                self.attention_weights[f'layer_{i}'] = {
                    'weights': self.layers[i]._alpha,
                    'edge_index': edge_index_dict
                }
            
            # 对每个节点类型进行后处理
            for node_type in h_dict.keys():
                if node_type in out_dict:
                    h_new = out_dict[node_type]
                    if node_type == 'super':
                        # 确保super节点的输出维度正确
                        h_new = h_new.view(batch_size, -1)
                    if node_type in h_dict:
                        h_new = h_new + h_dict[node_type]  # 残差连接
                    h_dict_new[node_type] = h_new
                else:
                    h_dict_new[node_type] = h_dict[node_type]
            
            # 记录每层的统计信息
            self.layer_outputs[f'layer_{i}'] = self.get_layer_statistics(h_dict_new, i)
            
            h_dict = h_dict_new
        
        # 使用super节点的特征进行分类
        super_features = h_dict['super']  # [batch_size, hidden_dim]
        
        # 分类
        outputs = self.classifier(super_features)
        
        return outputs

    def get_attention_visualization(self, layer_idx: int) -> Dict:
        """获取指定层的注意力权重可视化数据"""
        if f'layer_{layer_idx}' not in self.attention_weights:
            return None
            
        layer_data = self.attention_weights[f'layer_{layer_idx}']
        weights = layer_data['weights']
        edge_index = layer_data['edge_index']
        
        # 计算每种边类型的平均注意力权重
        attention_stats = {}
        for edge_type, edges in edge_index.items():
            if weights is not None:
                attention_stats[str(edge_type)] = {
                    'mean_attention': weights.mean().item(),
                    'max_attention': weights.max().item(),
                    'num_edges': edges.size(1)
                }
        
        return attention_stats

def calculate_metrics(outputs: torch.Tensor, labels: torch.Tensor) -> Dict:
    """计算详细的分类指标"""
    probabilities = torch.sigmoid(outputs)
    predictions = (probabilities > 0.5).float()
    
    metrics = {}
    tasks = ['bullying', 'aggression']
    
    for i, task in enumerate(tasks):
        tp = ((predictions[:, i] == 1) & (labels[:, i] == 1)).sum().item()
        fp = ((predictions[:, i] == 1) & (labels[:, i] == 0)).sum().item()
        tn = ((predictions[:, i] == 0) & (labels[:, i] == 0)).sum().item()
        fn = ((predictions[:, i] == 0) & (labels[:, i] == 1)).sum().item()
        
        eps = 1e-7
        accuracy = (tp + tn) / (tp + tn + fp + fn + eps)
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2 * (precision * recall) / (precision + recall + eps)
        
        # 计算更多指标
        specificity = tn / (tn + fp + eps)  # 特异度
        npv = tn / (tn + fn + eps)  # 负预测值
        fpr = fp / (fp + tn + eps)  # 假阳性率
        fnr = fn / (fn + tp + eps)  # 假阴性率
        
        # 计算AUC-ROC
        try:
            from sklearn.metrics import roc_auc_score, average_precision_score
            y_true = labels[:, i].cpu().numpy()
            y_score = probabilities[:, i].detach().cpu().numpy()
            auc_score = roc_auc_score(y_true, y_score)
            ap_score = average_precision_score(y_true, y_score)  # 平均精度分数
        except (ImportError, ValueError) as e:
            auc_score = 0.5
            ap_score = 0.5
            print(f"计算AUC/AP时出错: {e}")
        
        metrics[task] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'specificity': specificity,
            'npv': npv,
            'fpr': fpr,
            'fnr': fnr,
            'auc': auc_score,
            'ap': ap_score,
            'confusion_matrix': {
                'true_positive': tp,
                'false_positive': fp,
                'true_negative': tn,
                'false_negative': fn
            }
        }
        
        # 添加阈值无关的指标
        thresholds = torch.linspace(0, 1, 100)
        precisions = []
        recalls = []
        for threshold in thresholds:
            pred = (probabilities[:, i] > threshold).float()
            tp_t = ((pred == 1) & (labels[:, i] == 1)).sum().item()
            fp_t = ((pred == 1) & (labels[:, i] == 0)).sum().item()
            fn_t = ((pred == 0) & (labels[:, i] == 1)).sum().item()
            
            p = tp_t / (tp_t + fp_t + eps)
            r = tp_t / (tp_t + fn_t + eps)
            precisions.append(p)
            recalls.append(r)
            
        metrics[task]['pr_curve'] = {
            'precisions': precisions,
            'recalls': recalls,
            'thresholds': thresholds.tolist()
        }
    
    return metrics