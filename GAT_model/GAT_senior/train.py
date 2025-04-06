import torch
import torch.nn as nn
from typing import List, Dict, Optional
from data_loader import DataLoader
from HGT_model import HGTransformer, calculate_metrics
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time
import copy
from datetime import datetime
import random

def batch_generator(ids: List[str], batch_size: int):
    """生成批次数据的迭代器"""
    for i in range(0, len(ids), batch_size):
        yield ids[i:i + batch_size]

def calculate_metrics(outputs: torch.Tensor, labels: torch.Tensor) -> Dict:
    """计算分类指标"""
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
        
        auc_score = 0.5
        if tp + fn > 0 and tn + fp > 0:
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

def train_model(model: HGTransformer,
               data_loader: DataLoader,
               train_ids: List[str],
               val_ids: List[str],
               num_epochs: int = 20,
               lr: float = 0.001,
               batch_size: int = 64):
    device = next(model.parameters()).device
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
    # 添加学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=3,
        verbose=True,
        min_lr=1e-6
    )
    
    # 训练监控指标
    training_stats = {
        'train_losses': [],
        'val_losses': [],
        'learning_rates': [],
        'feature_stats': [],
        'attention_weights': [],
        'metrics': []
    }
    
    best_val_loss = float('inf')
    best_val_metrics = None
    
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        batch_count = 0
        epoch_feature_stats = []
        epoch_attention_weights = []
        
        # 训练阶段
        random.shuffle(train_ids)
        for i in range(0, len(train_ids), batch_size):
            batch_ids = train_ids[i:i + batch_size]
            batch_data = data_loader.load_batch(batch_ids)
            
            if batch_data is None:
                continue
                
            optimizer.zero_grad()
            
            outputs = model(batch_data['x_dict'], batch_data['edge_index_dict'])
            loss = criterion(outputs, batch_data['labels'])
            
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_train_loss += loss.item()
            batch_count += 1
            
            # 收集特征和注意力统计
            epoch_feature_stats.append(model.feature_stats)
            epoch_attention_weights.append(model.attention_weights)
            
            # 打印每个batch的学习率
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_count}, LR: {current_lr:.6f}, Loss: {loss.item():.4f}")
            
            # 打印注意力分布
            for layer_idx in range(model.num_layers):
                attention_vis = model.get_attention_visualization(layer_idx)
                if attention_vis:
                    print(f"\nLayer {layer_idx} Attention Distribution:")
                    for edge_type, stats in attention_vis.items():
                        print(f"  {edge_type}:")
                        print(f"    Mean Attention: {stats['mean_attention']:.4f}")
                        print(f"    Max Attention: {stats['max_attention']:.4f}")
                        print(f"    Edge Count: {stats['num_edges']}")
        
        avg_train_loss = total_train_loss / batch_count if batch_count > 0 else float('inf')
        
        # 验证阶段
        model.eval()
        total_val_loss = 0
        val_batch_count = 0
        all_val_outputs = []
        all_val_labels = []
        
        with torch.no_grad():
            for i in range(0, len(val_ids), batch_size):
                batch_ids = val_ids[i:i + batch_size]
                batch_data = data_loader.load_batch(batch_ids)
                
                if batch_data is None:
                    continue
                    
                outputs = model(batch_data['x_dict'], batch_data['edge_index_dict'])
                loss = criterion(outputs, batch_data['labels'])
                
                total_val_loss += loss.item()
                val_batch_count += 1
                
                all_val_outputs.append(outputs)
                all_val_labels.append(batch_data['labels'])
        
        avg_val_loss = total_val_loss / val_batch_count if val_batch_count > 0 else float('inf')
        
        # 计算验证集的详细指标
        val_outputs = torch.cat(all_val_outputs, dim=0)
        val_labels = torch.cat(all_val_labels, dim=0)
        val_metrics = calculate_metrics(val_outputs, val_labels)
        
        # 更新学习率
        scheduler.step(avg_val_loss)
        
        # 保存训练统计信息
        training_stats['train_losses'].append(avg_train_loss)
        training_stats['val_losses'].append(avg_val_loss)
        training_stats['learning_rates'].append(optimizer.param_groups[0]['lr'])
        training_stats['feature_stats'].append(epoch_feature_stats)
        training_stats['attention_weights'].append(epoch_attention_weights)
        training_stats['metrics'].append(val_metrics)
        
        # 打印详细的训练信息
        print(f"\nEpoch {epoch+1}/{num_epochs} Summary:")
        print(f"Average Train Loss: {avg_train_loss:.4f}")
        print(f"Average Validation Loss: {avg_val_loss:.4f}")
        print("\nValidation Metrics:")
        for task, metrics in val_metrics.items():
            print(f"\n{task.upper()}:")
            print(f"F1 Score: {metrics['f1']:.4f}")
            print(f"AUC-ROC: {metrics['auc']:.4f}")
            print(f"Precision: {metrics['precision']:.4f}")
            print(f"Recall: {metrics['recall']:.4f}")
        
        # 更新最佳模型状态
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_val_metrics = val_metrics
            print("\nNew best model found!")
        
        print("\n" + "="*50 + "\n")
    
    # 返回训练统计信息和最佳验证指标
    return model, training_stats, best_val_metrics

def log_predictions(outputs, labels, log_file):
    """记录预测结果"""
    probabilities = torch.sigmoid(outputs)
    predictions = (probabilities > 0.5).float()
    
    for i in range(len(outputs)):
        log_file.write(f"[{probabilities[i][0]:.4f}, {probabilities[i][1]:.4f}] -> ")
        log_file.write(f"[{predictions[i][0]:.0f}, {predictions[i][1]:.0f}] | ")
        log_file.write(f"[{labels[i][0]:.0f}, {labels[i][1]:.0f}]\n")

def log_metrics(metrics, log_file):
    """记录评估指标"""
    for task in ['bullying', 'aggression']:
        log_file.write(f"\n--- {task}检测指标 ---\n")
        log_file.write(f"Accuracy: {metrics[task]['accuracy']:.4f}\n")
        log_file.write(f"Precision: {metrics[task]['precision']:.4f}\n")
        log_file.write(f"Recall: {metrics[task]['recall']:.4f}\n")
        log_file.write(f"F1 Score: {metrics[task]['f1']:.4f}\n")
        log_file.write(f"AUC: {metrics[task]['auc']:.4f}\n")

def evaluate_model(model, data_loader, val_ids, batch_size, criterion, log_file, epoch):
    """评估模型性能"""
    val_loss = 0
    val_batch_count = 0
    all_outputs = []
    all_labels = []
    
    with torch.no_grad():
        for batch_idx, batch_ids in enumerate(batch_generator(val_ids, batch_size)):
            batch_data = data_loader.load_batch(batch_ids)
            
            if batch_data is None:
                continue
            
            try:
                val_batch_count += 1
                
                # 准备输入数据
                x_dict = {k: v.to(next(model.parameters()).device) for k, v in batch_data['x_dict'].items()}
                edge_index_dict = {k: v.to(next(model.parameters()).device) for k, v in batch_data['edge_index_dict'].items()}
                batch_labels = batch_data['labels'].to(next(model.parameters()).device)
                
                # 前向传播
                outputs = model(x_dict, edge_index_dict)
                
                # 确保标签形状与输出匹配
                if outputs.shape != batch_labels.shape:
                    outputs = outputs.view(batch_labels.shape)
                
                # 计算损失
                loss = criterion(outputs, batch_labels)
                val_loss += loss.item()
                
                # 记录验证集预测
                log_file.write(f"\n=== Epoch {epoch}, Validation Batch {batch_idx} ===\n")
                log_predictions(outputs, batch_labels, log_file)
                
                all_outputs.append(outputs)
                all_labels.append(batch_labels)
            
            except RuntimeError as e:
                print(f"\nError in validation batch {val_batch_count}: {str(e)}")
                print("跳过此批次...")
                continue
    
    avg_val_loss = val_loss / max(1, val_batch_count)
    
    # 计算指标
    if all_outputs and all_labels:
        all_outputs = torch.cat(all_outputs, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        metrics = calculate_metrics(all_outputs, all_labels)
        
        # 打印验证信息
        log_file.write(f"\n--- Epoch {epoch} Validation Summary ---\n")
        log_file.write(f"验证损失: {avg_val_loss:.4f}\n")
        log_file.write(f"验证批次数: {val_batch_count}\n")
        
        # 记录验证指标
        log_metrics(metrics, log_file)
    else:
        log_file.write(f"Epoch {epoch} - Val Loss: {avg_val_loss:.4f}\n")
        log_file.write("警告: 没有收集到有效的验证集输出和标签\n")
    
    return metrics
    