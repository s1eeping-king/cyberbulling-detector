import torch
import torch.nn as nn
from typing import List
from data_loader import DataLoader
from GAT_v1 import HeteroGAT, calculate_metrics

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