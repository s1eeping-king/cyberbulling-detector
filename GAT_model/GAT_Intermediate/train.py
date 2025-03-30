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
               lr: float = 0.001,
               debug: bool = False):
    """训练模型"""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # 获取设备
    device = next(model.parameters()).device
    
    # 先计算类别权重
    print("\n计算类别权重...")
    pos_samples = torch.zeros(2, device=device)
    neg_samples = torch.zeros(2, device=device)
    valid_samples = 0
    
    # 正确计算每个类别的正负样本数
    sample_count = 0
    for media_id in train_ids:
        labels = data_loader.get_labels(media_id)
        if labels is not None:
            pos_samples += (labels == 1).float()
            neg_samples += (labels == 0).float()
            valid_samples += 1
            sample_count += 1
            # 只显示前3个样本的标签
            if sample_count <= 3:
                print(f"\n样本 {sample_count} 标签: {labels.cpu().numpy()}")
    
    if valid_samples > 0:
        print(f"\n数据统计 (总样本数: {valid_samples}):")
        print(f"Bullying - 正样本: {pos_samples[0].item():.0f}, 负样本: {neg_samples[0].item():.0f}")
        print(f"Aggression - 正样本: {pos_samples[1].item():.0f}, 负样本: {neg_samples[1].item():.0f}")
        
        # 确保不会出现除零错误
        pos_samples = torch.max(pos_samples, torch.ones_like(pos_samples))
        pos_weight = neg_samples / pos_samples
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        print(f"类别权重 - Bullying: {pos_weight[0].item():.2f}, Aggression: {pos_weight[1].item():.2f}")
    else:
        criterion = nn.BCEWithLogitsLoss()
        print("使用默认权重")
    
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    
    print("\nStarting training...")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        valid_batches = 0
        
        for media_id in train_ids:
            # 加载子图和标签
            data = data_loader.load_subgraph(media_id)
            labels = data_loader.get_labels(media_id)
            
            if data is None or labels is None:
                continue
                
            valid_batches += 1
            
            # 准备输入数据
            x_dict = {node_type: data[node_type].x for node_type in data.node_types}
            edge_index_dict = {edge_type: data[edge_type].edge_index for edge_type in data.edge_types}
            
            # 前向传播
            outputs = model(x_dict, edge_index_dict)
            
            # 确保标签形状与输出匹配
            if outputs.shape[0] != 1:
                outputs = outputs.mean(dim=0, keepdim=True)
            labels = labels.view(1, -1)
            
            # 只在第一个epoch的前2个batch显示详细信息
            if epoch == 0 and valid_batches < 2:
                print(f"\nBatch {valid_batches} - 输出: {torch.sigmoid(outputs).detach().cpu().numpy().round(3)}, 标签: {labels.cpu().numpy()}")
            
            # 计算损失
            loss = criterion(outputs, labels)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_train_loss = total_loss / max(1, valid_batches)
        
        # 验证阶段
        model.eval()
        val_loss = 0
        val_batch_count = 0
        all_outputs = []
        all_labels = []
        
        with torch.no_grad():
            for media_id in val_ids:
                # 加载子图和标签
                data = data_loader.load_subgraph(media_id)
                labels = data_loader.get_labels(media_id)
                
                if data is None or labels is None:
                    continue
                    
                # 准备输入数据
                x_dict = {node_type: data[node_type].x for node_type in data.node_types}
                edge_index_dict = {edge_type: data[edge_type].edge_index for edge_type in data.edge_types}
                
                # 前向传播
                outputs = model(x_dict, edge_index_dict)
                
                # 确保标签形状与输出匹配
                if outputs.shape[0] != 1:
                    outputs = outputs.mean(dim=0, keepdim=True)
                labels = labels.view(1, -1)
                
                # 计算损失
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                val_batch_count += 1
                
                all_outputs.append(outputs)
                all_labels.append(labels)
        
        avg_val_loss = val_loss / max(1, val_batch_count)
        
        # 计算指标
        if all_outputs and all_labels:
            all_outputs = torch.cat(all_outputs, dim=0)
            all_labels = torch.cat(all_labels, dim=0)
            metrics = calculate_metrics(all_outputs, all_labels)
            
            # 打印详细的评估指标
            print(f"\n{'='*20} 评估指标 {'='*20}")
            print(f"Epoch {epoch+1}/{num_epochs}{'='*40}")
            print(f"训练损失: {avg_train_loss:.4f}, 验证损失: {avg_val_loss:.4f}")
            
            # 霸凌检测指标
            print(f"\n--- 霸凌检测指标 ---")
            print(f"准确率 (Accuracy): {metrics['bullying']['accuracy']:.4f}")
            print(f"精确率 (Precision): {metrics['bullying']['precision']:.4f}")
            print(f"召回率 (Recall): {metrics['bullying']['recall']:.4f}")
            print(f"F1分数 (F1 Score): {metrics['bullying']['f1']:.4f}")
            if 'auc' in metrics['bullying']:
                print(f"AUC-ROC: {metrics['bullying']['auc']:.4f}")
            
            # 攻击性检测指标
            print(f"\n--- 攻击性检测指标 ---")
            print(f"准确率 (Accuracy): {metrics['aggression']['accuracy']:.4f}")
            print(f"精确率 (Precision): {metrics['aggression']['precision']:.4f}")
            print(f"召回率 (Recall): {metrics['aggression']['recall']:.4f}")
            print(f"F1分数 (F1 Score): {metrics['aggression']['f1']:.4f}")
            if 'auc' in metrics['aggression']:
                print(f"AUC-ROC: {metrics['aggression']['auc']:.4f}")
            
            print(f"Epoch {epoch+1}/{num_epochs} - "
                  f"Train Loss: {avg_train_loss:.4f}, "
                  f"Val Loss: {avg_val_loss:.4f}, "
                  f"Bullying F1: {metrics['bullying']['f1']:.4f}, "
                  f"Aggression F1: {metrics['aggression']['f1']:.4f}")
        else:
            print(f"Epoch {epoch+1}/{num_epochs} - "
                  f"Train Loss: {avg_train_loss:.4f}, "
                  f"Val Loss: {avg_val_loss:.4f}")
            print("警告: 没有收集到有效的验证集输出和标签")
        
        # 早停
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    return model
    