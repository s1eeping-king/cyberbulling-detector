import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, HeteroConv
from torch_geometric.data import HeteroData
from typing import Dict, List, Tuple
from data_loader import DataLoader
import logging
import random

class HeteroGAT(nn.Module):
    """异构图注意力网络"""
    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        # 节点特征维度
        self.feature_dims = {
            'user': 6,          # [followerCount, followingCount, likeCount, postCount, username_len, description_len]
            'media_session': 8,  # [likeCount, commentCount, loopCount, repostCount, desc_len, emotion_conf, theme_conf, time]
            'comment': 2,       # [text_length, postId_hash]
            'label': 4         # [is_bullying, is_aggression, bullying_conf, aggression_conf]
        }
        
        # 边的类型
        self.edge_types = [
            ('user', 'publishes', 'media_session'),
            ('user', 'creates', 'comment'),
            ('comment', 'mentions', 'user'),
            ('media_session', 'has_label', 'label'),
            ('comment', 'belongs_to', 'media_session')
        ]
        
        # 特征转换层
        self.encoders = nn.ModuleDict({
            node_type: nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            )
            for node_type, dim in self.feature_dims.items()
        })
        
        # GAT层
        self.conv = HeteroConv({
            edge_type: GATConv(hidden_dim, hidden_dim, add_self_loops=False)
            for edge_type in self.edge_types
        })
        
        # 预测层
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 2)  # 二分类：霸凌和攻击性
        )
        
    def forward(self, x_dict, edge_index_dict):
        """前向传播"""
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
        
        # 2. 消息传递
        h_dict = self.conv(h_dict, edge_index_dict)
        
        # 3. 预测（仅对媒体会话节点）
        return self.classifier(h_dict['media_session'])

def calculate_metrics(outputs, labels):
    """计算分类指标"""
    predictions = (torch.sigmoid(outputs) > 0.5).float()
    
    # 计算每个类别的指标
    metrics = {}
    for i, task in enumerate(['bullying', 'aggression']):
        tp = ((predictions[:, i] == 1) & (labels[:, i] == 1)).sum().item()
        fp = ((predictions[:, i] == 1) & (labels[:, i] == 0)).sum().item()
        tn = ((predictions[:, i] == 0) & (labels[:, i] == 0)).sum().item()
        fn = ((predictions[:, i] == 0) & (labels[:, i] == 1)).sum().item()
        
        # 避免除零
        eps = 1e-7
        accuracy = (tp + tn) / (tp + tn + fp + fn + eps)
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2 * (precision * recall) / (precision + recall + eps)
        
        metrics[task] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    return metrics

def train_model(model: HeteroGAT,
               data_loader: DataLoader,
               train_ids: List[str],
               val_ids: List[str],
               num_epochs: int = 10,
               lr: float = 0.001):
    """训练模型"""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # 获取设备
    device = next(model.parameters()).device
    
    # 先计算类别权重
    print("\nCalculating class weights...")
    pos_count = torch.zeros(2, device=device)
    total_count = 0
    
    for media_id in train_ids:
        data = data_loader.load_subgraph(media_id)
        if data is not None:
            pos_count += data['media_session'].y.sum(dim=0)
            total_count += 1
            
    if total_count > 0:
        neg_count = total_count - pos_count
        pos_weight = neg_count / pos_count
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        print(f"Class weights: {pos_weight}")
    else:
        criterion = nn.BCEWithLogitsLoss()
        print("Using default weights")
    
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    
    print("\nStarting training...")
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        total_loss = 0
        batch_count = 0
        train_outputs = []
        train_labels = []
        
        for media_id in train_ids:
            data = data_loader.load_subgraph(media_id)
            if data is None:
                continue
                
            optimizer.zero_grad()
            out = model(data.x_dict, data.edge_index_dict)
            loss = criterion(out, data['media_session'].y)
            
            # 收集预测和标签
            train_outputs.append(out.detach())
            train_labels.append(data['media_session'].y.detach())
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            batch_count += 1
            
        avg_train_loss = total_loss / batch_count if batch_count > 0 else 0
        
        # 计算训练集指标
        if train_outputs:
            train_outputs = torch.cat(train_outputs, dim=0)
            train_labels = torch.cat(train_labels, dim=0)
            train_metrics = calculate_metrics(train_outputs, train_labels)
        
        # 验证阶段
        model.eval()
        val_loss = 0
        val_batch_count = 0
        val_outputs = []
        val_labels = []
        
        with torch.no_grad():
            for media_id in val_ids:
                data = data_loader.load_subgraph(media_id)
                if data is None:
                    continue
                    
                out = model(data.x_dict, data.edge_index_dict)
                loss = criterion(out, data['media_session'].y)
                
                # 收集预测和标签
                val_outputs.append(out)
                val_labels.append(data['media_session'].y)
                
                val_loss += loss.item()
                val_batch_count += 1
                
        avg_val_loss = val_loss / val_batch_count if val_batch_count > 0 else 0
        
        # 计算验证集指标
        if val_outputs:
            val_outputs = torch.cat(val_outputs, dim=0)
            val_labels = torch.cat(val_labels, dim=0)
            val_metrics = calculate_metrics(val_outputs, val_labels)
        
        # 打印训练信息
        print(f'\nEpoch {epoch+1:02d}/{num_epochs:02d}:')
        print(f'  Train - Loss: {avg_train_loss:.4f}')
        for task in ['bullying', 'aggression']:
            print(f'    {task.capitalize():9s} - Acc: {train_metrics[task]["accuracy"]:.4f}, '
                  f'Prec: {train_metrics[task]["precision"]:.4f}, '
                  f'Rec: {train_metrics[task]["recall"]:.4f}, '
                  f'F1: {train_metrics[task]["f1"]:.4f}')
        
        print(f'  Val   - Loss: {avg_val_loss:.4f}')
        for task in ['bullying', 'aggression']:
            print(f'    {task.capitalize():9s} - Acc: {val_metrics[task]["accuracy"]:.4f}, '
                  f'Prec: {val_metrics[task]["precision"]:.4f}, '
                  f'Rec: {val_metrics[task]["recall"]:.4f}, '
                  f'F1: {val_metrics[task]["f1"]:.4f}')
        
        # 早停
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print('\nEarly stopping!')
                break
                
    print("\nTraining finished.")

def main():
    # 设置随机种子
    random.seed(42)
    torch.manual_seed(42)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 加载Neo4j连接信息
    with open("knowledge_graph/neo4j_information.txt", 'r') as f:
        neo4j_info = {}
        for line in f:
            key, value = line.strip().split(' = ')
            neo4j_info[key] = value.strip('"')
    
    # 初始化模型和数据加载器
    model = HeteroGAT().to(device)
    data_loader = DataLoader(
        uri=neo4j_info['uri'],
        user=neo4j_info['username'],
        password=neo4j_info['password'],
        device=device
    )
    
    try:
        # 获取所有媒体会话ID并划分训练集和验证集
        all_ids = data_loader.get_all_media_sessions()
        random.shuffle(all_ids)  # 随机打乱数据
        split = int(len(all_ids) * 0.8)
        train_ids = all_ids[:split]
        val_ids = all_ids[split:]
        
        print(f"Total samples: {len(all_ids)}")
        print(f"Training samples: {len(train_ids)}")
        print(f"Validation samples: {len(val_ids)}")
        
        # 训练模型
        train_model(model, data_loader, train_ids, val_ids)
        
    finally:
        # 确保关闭数据库连接
        data_loader.close()

if __name__ == "__main__":
    main()
