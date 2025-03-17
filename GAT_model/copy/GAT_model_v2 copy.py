import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.data import Data, Batch
import numpy as np
import logging
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, List, Optional
import pickle
import os
from torch_geometric.loader import DataLoader

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 焦点损失函数实现
class FocalLoss(nn.Module):
    """焦点损失函数，用于处理类别不平衡问题"""
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.eps = 1e-6

    def forward(self, inputs, targets):
        # 转换为概率
        probs = F.softmax(inputs, dim=1)
        # 获取目标类别的概率
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        # 计算焦点权重
        loss = -self.alpha * (1 - pt) ** self.gamma * torch.log(pt + self.eps)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

# 残差连接模块
class ResidualConnection(nn.Module):
    """残差连接模块，用于改善梯度流动"""
    def __init__(self, in_channels, out_channels):
        super(ResidualConnection, self).__init__()
        self.lin = nn.Linear(in_channels, out_channels) if in_channels != out_channels else nn.Identity()
    
    def forward(self, x_in, x_out):
        return x_out + self.lin(x_in)

class EnhancedGATConv(nn.Module):
    """增强版GAT卷积层，添加了残差连接和多头注意力"""
    def __init__(self, in_channels, out_channels, heads=4, dropout=0.2, residual=True):
        super(EnhancedGATConv, self).__init__()
        self.gat = GATConv(in_channels, out_channels, heads=heads, dropout=dropout)
        self.residual = residual
        if residual:
            self.res_conn = ResidualConnection(in_channels, out_channels * heads)
    
    def forward(self, x, edge_index):
        x_in = x
        x = self.gat(x, edge_index)
        if self.residual:
            x = self.res_conn(x_in, x)
        return x

class MultiScalePooling(nn.Module):
    """多尺度池化层，结合多种池化方式捕获不同粒度的信息"""
    def __init__(self, in_channels):
        super(MultiScalePooling, self).__init__()
        self.in_channels = in_channels
        self.proj = nn.Linear(in_channels * 3, in_channels)
    
    def forward(self, x, batch):
        # 应用不同的池化方法
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x_sum = global_add_pool(x, batch)
        
        # 拼接并投影回原始维度
        x_cat = torch.cat([x_mean, x_max, x_sum], dim=1)
        return self.proj(x_cat)

class NodeEmbedding(nn.Module):
    """节点嵌入模块，用于学习图结构特征"""
    def __init__(self, num_nodes, embedding_dim=128):
        super(NodeEmbedding, self).__init__()
        self.embedding = nn.Embedding(num_nodes, embedding_dim)
    
    def forward(self, node_ids):
        return self.embedding(node_ids)

class ImprovedCyberGAT(nn.Module):
    """
    改进版基于图注意力网络(GAT)的网络霸凌检测模型
    添加了残差连接、多尺度特征融合和自注意力机制
    """
    def __init__(self, in_channels, hidden_channels=256, num_heads=4, num_layers=3, dropout=0.2, edge_dim=None):
        super(ImprovedCyberGAT, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.edge_dim = edge_dim

        # GAT层
        self.gat_layers = nn.ModuleList()
        self.residual_connections = nn.ModuleList()
        
        # 第一层
        self.gat_layers.append(
            EnhancedGATConv(in_channels, hidden_channels, heads=num_heads, dropout=dropout)
        )
        
        # 中间层
        for _ in range(num_layers - 2):
            self.gat_layers.append(
                EnhancedGATConv(hidden_channels * num_heads, hidden_channels, heads=num_heads, dropout=dropout)
            )
        
        # 最后一层（输出层）
        self.gat_layers.append(
            EnhancedGATConv(hidden_channels * num_heads, hidden_channels, heads=1, dropout=dropout)
        )
        
        # 批归一化层
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_channels * num_heads) for _ in range(num_layers - 1)
        ] + [nn.BatchNorm1d(hidden_channels)])
        
        # 多尺度池化层
        self.pooling = MultiScalePooling(hidden_channels)
        
        # 自注意力层 - 用于节点特征加权
        self.self_attention = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 4),
            nn.LeakyReLU(),
            nn.Linear(hidden_channels // 4, 1)
        )
        
        # 预测头 - 多层感知机
        self.mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, 2)  # 二分类：霸凌和非霸凌
        )

    def forward(self, x, edge_index, batch=None, edge_attr=None):
        """
        前向传播
        Args:
            x: 节点特征矩阵 [num_nodes, in_channels]
            edge_index: 边索引 [2, num_edges]
            batch: 批处理索引 [num_nodes]
            edge_attr: 边特征 [num_edges, edge_dim]
        Returns:
            out: 预测结果 [num_nodes, 2]
        """
        # 保存中间特征用于残差连接
        prev_x = x
        
        # GAT层
        for i in range(self.num_layers):
            x = self.gat_layers[i](x, edge_index)
            x = self.batch_norms[i](x)
            if i < self.num_layers - 1:  # 除最后一层外都使用激活函数和Dropout
                x = F.leaky_relu(x, negative_slope=0.2)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        # 应用自注意力机制
        if batch is not None:
            # 计算注意力权重
            attn_weights = self.self_attention(x).squeeze(-1)
            attn_weights = F.softmax(attn_weights, dim=0)
            
            # 使用多尺度池化获取全局信息
            x = self.pooling(x, batch)
        
        # 预测
        out = self.mlp(x)
        
        return out

class CyberbullyingDetector:
    """
    网络霸凌检测器
    整合知识图谱和改进版GAT模型
    """
    def __init__(self, in_channels, device=None, edge_dim=None):
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = ImprovedCyberGAT(in_channels=in_channels, edge_dim=edge_dim).to(self.device)
        # 计算类别权重
        self.criterion = None  # 将在prepare_graph_data中设置
        self.focal_loss = None  # 将在prepare_graph_data中设置
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.001, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=0.5, patience=5)
        
        logger.info(f"Initialized CyberbullyingDetector on device: {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters())}")

    def prepare_graph_data(self, features_list, edge_index_list, labels_list, edge_attr_list=None):
        """
        准备图数据
        """
        data_list = []
        for i, (features, edge_index, labels) in enumerate(zip(features_list, edge_index_list, labels_list)):
            logger.info(f"Preparing graph {i} with {len(features)} nodes and {edge_index.shape[1]} edges")
            
            # 检查特征有效性
            if len(features) == 0:
                logger.warning(f"Empty features for graph {i}, skipping...")
                continue
            
            # 确保边索引在有效范围内
            mask = (edge_index[0] < len(features)) & (edge_index[1] < len(features))
            edge_index = edge_index[:, mask]
            
            # 检查过滤后的边是否有效
            if edge_index.size == 0:
                logger.info(f"Creating minimum spanning tree for graph {i}...")
                # 创建最小生成树连接
                edge_index = np.array([[j, j+1] for j in range(len(features)-1)]).T
            
            logger.info(f"Graph {i} after filtering: {len(features)} nodes, {edge_index.shape[1]} edges")
            
            # 重新映射节点索引以确保连续性
            all_nodes = np.arange(len(features))
            node_idx_map = {old_idx: new_idx for new_idx, old_idx in enumerate(all_nodes)}
            new_edge_index = np.array([[node_idx_map[idx] for idx in edge_index[0]],
                                    [node_idx_map[idx] for idx in edge_index[1]]])
            
            # 准备特征和标签
            new_features = features
            new_labels = labels
            
            # 计算类别权重
            label_counts = np.bincount(new_labels)
            total_samples = len(new_labels)
            weights = torch.FloatTensor([total_samples / (len(label_counts) * count) for count in label_counts]).to(self.device)
            
            # 设置损失函数 - 使用交叉熵和焦点损失的组合
            self.criterion = nn.CrossEntropyLoss(weight=weights)
            self.focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
            
            # 转换为PyTorch张量并移动到正确的设备
            data_dict = {
                'x': torch.FloatTensor(new_features).to(self.device),
                'edge_index': torch.LongTensor(new_edge_index).to(self.device),
                'y': torch.LongTensor(new_labels).to(self.device),
                'num_nodes': len(new_features)
            }
            
            # 如果有边属性，添加到数据中
            if edge_attr_list is not None and i < len(edge_attr_list) and edge_attr_list[i] is not None:
                edge_attr = edge_attr_list[i]
                # 确保边属性与边索引匹配
                if edge_attr.shape[0] != edge_index.shape[1]:
                    logger.warning(f"Edge attributes shape {edge_attr.shape} doesn't match edge index shape {edge_index.shape}")
                    # 如果不匹配，创建默认边属性
                    edge_attr = np.ones((edge_index.shape[1], 1))
                data_dict['edge_attr'] = torch.FloatTensor(edge_attr).to(self.device)
            
            data = Data(**data_dict)
            data_list.append(data)
            logger.info(f"Successfully created graph data {i}")
        
        if not data_list:
            raise ValueError("No valid graph data found after filtering. Please check input data and edge filtering logic.")
            
        # 返回单个数据对象而不是列表
        return data_list

    def train_epoch(self, data):
        """
        训练一个epoch
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # 前向传播
        logger.debug(f"Training on data with x shape: {data.x.shape}, edge_index shape: {data.edge_index.shape}, batch shape: {data.batch.shape}")
        
        # 检查是否有边属性
        edge_attr = data.edge_attr if hasattr(data, 'edge_attr') else None
        out = self.model(data.x, data.edge_index, data.batch, edge_attr)
        
        # 结合交叉熵损失和焦点损失
        ce_loss = self.criterion(out, data.y)
        focal_loss = self.focal_loss(out, data.y)
        loss = ce_loss + 0.5 * focal_loss  # 权重可调整
        
        # 反向传播
        loss.backward()
        
        # 梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        return loss.item()

    def evaluate(self, data, batch_size=32):
        """
        评估模型
        """
        self.model.eval()
        try:
            # 验证数据的完整性
            if not hasattr(data, 'x') or not hasattr(data, 'edge_index') or not hasattr(data, 'y'):
                raise ValueError("数据缺少必要的属性(x, edge_index, y)")
            
            logger.info(f"评估数据信息 - 节点数: {data.num_nodes}, 边数: {data.edge_index.size(1)}")
            logger.info(f"数据特征维度 - x: {data.x.shape}, y: {data.y.shape}")
            logger.info(f"数据所在设备: {data.x.device}")
            
            # 直接使用整个数据进行评估，不使用DataLoader
            total_correct = 0
            total_samples = 0
            all_preds = []
            all_labels = []
            all_probs = []
            
            with torch.no_grad():
                # 确保数据在正确的设备上
                data = data.to(self.device)
                
                # 如果数据没有batch属性，添加一个全零的batch属性
                if not hasattr(data, 'batch'):
                    data.batch = torch.zeros(data.num_nodes, dtype=torch.long, device=self.device)
                
                # 检查是否有边属性
                edge_attr = data.edge_attr if hasattr(data, 'edge_attr') else None
                
                # 前向传播
                out = self.model(data.x, data.edge_index, data.batch, edge_attr)
                
                # 计算预测结果
                probs = F.softmax(out, dim=1)
                pred = out.argmax(dim=1)
                
                # 计算准确率
                total_correct = (pred == data.y).sum().item()
                total_samples = data.y.size(0)
                
                # 收集预测和标签
                all_preds = pred.cpu().numpy()
                all_labels = data.y.cpu().numpy()
                all_probs = probs.cpu().numpy()
                
                # 计算F1分数
                all_preds = np.array(all_preds)
                all_labels = np.array(all_labels)
                
                logger.info(f"预测结果统计 - 预测标签: {np.unique(all_preds, return_counts=True)}")
                logger.info(f"真实标签统计 - 真实标签: {np.unique(all_labels, return_counts=True)}")
                
                tp = ((all_preds == 1) & (all_labels == 1)).sum()
                fp = ((all_preds == 1) & (all_labels == 0)).sum()
                fn = ((all_preds == 0) & (all_labels == 1)).sum()
                tn = ((all_preds == 0) & (all_labels == 0)).sum()
                
                logger.info(f"性能指标 - TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}")
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                
                logger.info(f"最终性能指标 - Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
                
                return total_correct / total_samples, f1, all_probs
        except Exception as e:
            logger.error(f"评估过程中出现错误: {str(e)}")
            raise

    def train(self, train_data, val_data=None, epochs=100, batch_size=32, early_stopping=10):
        best_val_f1 = 0
        patience_counter = 0
        best_model_state = None
        
        # 确保数据在正确的设备上
        train_data = train_data.to(self.device)
        if val_data is not None:
            val_data = val_data.to(self.device)
        
        # 如果数据没有batch属性，添加一个全零的batch属性
        if not hasattr(train_data, 'batch'):
            train_data.batch = torch.zeros(train_data.num_nodes, dtype=torch.long, device=self.device)
        
        for epoch in range(epochs):
            # 训练
            self.model.train()
            self.optimizer.zero_grad()
            
            # 检查是否有边属性
            edge_attr = train_data.edge_attr if hasattr(train_data, 'edge_attr') else None
            
            # 前向传播
            out = self.model(train_data.x, train_data.edge_index, train_data.batch, edge_attr)
            
            # 结合交叉熵损失和焦点损失
            ce_loss = self.criterion(out, train_data.y)
            focal_loss = self.focal_loss(out, train_data.y)
            loss = ce_loss + 0.5 * focal_loss  # 权重可调整
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            total_loss = loss.item()
            
            # 评估
            train_acc, train_f1, _ = self.evaluate(train_data, batch_size)
            if val_data is not None:
                val_acc, val_f1, _ = self.evaluate(val_data, batch_size)
                
                # 更新学习率调度器
                self.scheduler.step(val_f1)
                
                # 早停检查
                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    patience_counter = 0
                    # 保存最佳模型状态
                    best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping:
                        logger.info(f"Early stopping triggered at epoch {epoch}")
                        # 恢复最佳模型状态
                        if best_model_state is not None:
                            self.model.load_state_dict(best_model_state)
                        break
                
                logger.info(f"Epoch {epoch}: Loss={total_loss:.4f}, "
                          f"Train Acc={train_acc:.4f}, Train F1={train_f1:.4f}, "
                          f"Val Acc={val_acc:.4f}, Val F1={val_f1:.4f}, "
                          f"LR={self.optimizer.param_groups[0]['lr']:.6f}")
            else:
                logger.info(f"Epoch {epoch}: Loss={total_loss:.4f}, "
                          f"Train Acc={train_acc:.4f}, Train F1={train_f1:.4f}")

    def predict(self, data):
        """
        预测
        """
        self.model.eval()
        with torch.no_grad():
            # 确保数据在正确的设备上
            data = data.to(self.device)
            
            # 如果数据没有batch属性，添加一个全零的batch属性
            if not hasattr(data, 'batch'):
                data.batch = torch.zeros(data.num_nodes, dtype=torch.long, device=self.device)
            
            # 检查是否有边属性
            edge_attr = data.edge_attr if hasattr(data, 'edge_attr') else None
            
            # 前向传播
            out = self.model(data.x, data.edge_index, data.batch, edge_attr)
            return F.softmax(out, dim=1).cpu().numpy()

def extract_features_and_labels(G):
    """从知识图谱中提取特征和标签"""
    features_list = []
    labels_list = []
    node_mapping = {}  # 用于映射节点ID到索引
    current_idx = 0

    # 遍历所有媒体会话节点
    for node in G.nodes():
        if node.startswith('media_session_'):
            node_data = G.nodes[node]
            video_id = node_data.get('video_id')
            
            # 获取视频节点的特征
            video_node = f"video_{video_id}"
            if video_node not in G:
                continue
            
            # 获取标签
            bullying_label = None
            for _, label_node in G.out_edges(node):
                label_data = G.nodes[label_node]
                if (label_data.get('node_type') == 'Label' and 
                    label_data.get('label_type') == 'bullying'):
                    bullying_label = label_data.get('value')
                    break
            
            if bullying_label is not None:
                # 获取特征
                session_features = node_data.get('session_features')
                video_features = G.nodes[video_node].get('frame_features')
                
                # 确保特征存在且格式正确
                if session_features is not None and video_features is not None:
                    # 转换为numpy数组
                    if isinstance(session_features, torch.Tensor):
                        session_features = session_features.cpu().numpy()
                    if isinstance(video_features, torch.Tensor):
                        video_features = video_features.cpu().numpy()
                    
                    # 处理特征维度
                    if len(video_features.shape) > 1:
                        video_features = np.mean(video_features, axis=0)
                    if len(session_features.shape) > 1:
                        session_features = np.mean(session_features, axis=0)
                    
                    # 合并特征
                    combined_features = np.concatenate([video_features, session_features])
                    
                    features_list.append(combined_features)
                    labels_list.append(bullying_label)
                    node_mapping[node] = current_idx
                    current_idx += 1

    if not features_list:
        raise ValueError("No valid features found in knowledge graph")
        
    return np.array(features_list), np.array(labels_list), node_mapping

def create_graph_from_features(features, labels, k=5):
    """
    根据特征相似度构建图
    使用kNN方法构建边，并添加边权重
    """
    from sklearn.neighbors import kneighbors_graph
    from sklearn.metrics.pairwise import cosine_similarity
    
    # 构建kNN图
    adj_matrix = kneighbors_graph(features, n_neighbors=k, mode='connectivity')
    adj_matrix = adj_matrix.maximum(adj_matrix.transpose())  # 确保是无向图
    
    # 计算边权重（基于余弦相似度）
    edge_index = np.array(adj_matrix.nonzero())
    edge_weights = []
    
    for i in range(edge_index.shape[1]):
        src, dst = edge_index[0, i], edge_index[1, i]
        sim = cosine_similarity(features[src].reshape(1, -1), features[dst].reshape(1, -1))[0][0]
        edge_weights.append(sim)
    
    edge_weights = np.array(edge_weights).reshape(-1, 1)
    
    return edge_index, edge_weights

if __name__ == "__main__":
    # 加载知识图谱
    logger.info("Loading knowledge graph...")
    try:
        with open('knowledge_graph/knowledge_graph.pkl', 'rb') as f:
            G = pickle.load(f)
        logger.info("Knowledge graph loaded successfully")
    except Exception as e:
        logger.error(f"Error loading knowledge graph: {str(e)}")
        raise

    # 从知识图谱中提取特征和标签
    logger.info("Extracting features and labels...")
    features, labels, node_mapping = extract_features_and_labels(G)
    logger.info(f"Extracted {len(features)} samples with {features.shape[1]} features")
    logger.info(f"Label distribution: {np.bincount(labels)}")

    # 构建图
    logger.info("Creating graph from features...")
    edge_index, edge_weights = create_graph_from_features(features, labels, k=5)
    logger.info(f"Created graph with {edge_index.shape[1]} edges")

    # 划分数据集
    logger.info("Splitting dataset...")
    train_idx, test_idx = train_test_split(np.arange(len(features)), test_size=0.2, stratify=labels)
    train_idx, val_idx = train_test_split(train_idx, test_size=0.2, stratify=labels[train_idx])

    # 创建检测器实例
    detector = CyberbullyingDetector(in_channels=features.shape[1], edge_dim=edge_weights.shape[1])

    # 准备数据
    logger.info("Preparing data...")
    
    # 为每个子集创建独立的边索引和边属性
    def filter_edges(edge_index, edge_weights, node_indices):
        # 创建节点索引映射
        idx_map = {old_idx: new_idx for new_idx, old_idx in enumerate(node_indices)}
        
        # 找出在node_indices中的边
        mask = np.isin(edge_index[0], node_indices) & np.isin(edge_index[1], node_indices)
        filtered_edges = edge_index[:, mask]
        filtered_weights = edge_weights[mask]
        
        if filtered_edges.size == 0:
            # 如果没有边，创建最小生成树连接
            new_edges = []
            new_weights = []
            for i in range(len(node_indices)-1):
                new_edges.append([node_indices[i], node_indices[i+1]])
                # 默认边权重为1.0
                new_weights.append([1.0])
            filtered_edges = np.array(new_edges).T
            filtered_weights = np.array(new_weights)
        
        # 重映射节点索引
        remapped_edges = np.array([[idx_map[src] for src in filtered_edges[0]],
                                  [idx_map[dst] for dst in filtered_edges[1]]])
        
        return remapped_edges, filtered_weights
    
    train_edges, train_weights = filter_edges(edge_index, edge_weights, train_idx)
    val_edges, val_weights = filter_edges(edge_index, edge_weights, val_idx)
    test_edges, test_weights = filter_edges(edge_index, edge_weights, test_idx)
    
    # 检查边索引有效性
    if train_edges.size == 0:
        raise ValueError("训练集边索引为空，请尝试：\n1. 增大kNN参数k值\n2. 检查特征矩阵有效性\n3. 验证节点映射关系")
    
    # 准备数据集
    train_data = detector.prepare_graph_data([features[train_idx]], [train_edges], [labels[train_idx]], [train_weights])
    val_data = detector.prepare_graph_data([features[val_idx]], [val_edges], [labels[val_idx]], [val_weights])
    test_data = detector.prepare_graph_data([features[test_idx]], [test_edges], [labels[test_idx]], [test_weights])

    # 训练模型
    logger.info("Training model...")
    # 直接传递列表中的第一个元素，确保它是一个完整的Data对象
    if len(train_data) > 0 and len(val_data) > 0:
        detector.train(train_data[0], val_data[0], epochs=100, early_stopping=10)
    else:
        logger.error("训练或验证数据为空，无法进行训练")

    logger.info("Evaluating model...")
    detector.model.eval()
    with torch.no_grad():
        # 检查是否有边属性
        edge_attr = test_data[0].edge_attr if hasattr(test_data[0], 'edge_attr') else None
        
        test_out = detector.model(test_data[0].x, test_data[0].edge_index, test_data[0].batch, edge_attr)
        test_pred = test_out.argmax(dim=1).cpu().numpy()
        test_labels = test_data[0].y.cpu().numpy()
        
        # 确保有两个类别的预测
        unique_labels = np.unique(np.concatenate([test_pred, test_labels]))
        if len(unique_labels) < 2:
            logger.warning(f"Warning: Only found labels: {unique_labels}")
        
        # 打印分类报告
        logger.info("\nClassification Report:")
        print(classification_report(test_labels, test_pred, 
                                 labels=[0, 1],
                                 target_names=['Non-bullying', 'Bullying']))

    # 绘制混淆矩阵
    cm = confusion_matrix(test_labels, test_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('GAT_model/GAT_results/confusion_matrix.png')
    plt.close()

    # 保存模型
    logger.info("Saving model...")
    torch.save({
        'model_state_dict': detector.model.state_dict(),
        'feature_dim': features.shape[1],
        'edge_dim': edge_weights.shape[1] if edge_weights is not None else None,
        'node_mapping': node_mapping
    }, 'GAT_model/GAT_results/cyberbullying_detector.pt')

    logger.info("Training and evaluation completed successfully")
    logger.info("Model saved as 'cyberbullying_detector.pt'")
    logger.info("Confusion matrix saved as 'confusion_matrix.png'")
