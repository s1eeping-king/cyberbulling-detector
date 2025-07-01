import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import List
from data_loader import DataLoader
from GAT import SimpleGAT, calculate_metrics

class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance"""
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def create_balanced_batch(data_loader: DataLoader, ids: List[str], batch_size: int, target_pos_ratio: float = 0.6):
    """创建类别平衡的批次 - 增加正样本比例"""
    pos_ids = []
    neg_ids = []

    for id in ids:
        label = data_loader.get_label(id)
        if label == 1:
            pos_ids.append(id)
        else:
            neg_ids.append(id)

    # 增加正样本比例到60%
    target_pos_count = int(batch_size * target_pos_ratio)
    target_pos_count = min(target_pos_count, len(pos_ids))
    
    target_neg_count = batch_size - target_pos_count
    target_neg_count = min(target_neg_count, len(neg_ids))
    
    actual_batch_size = target_pos_count + target_neg_count

    batch_pos_ids = np.random.choice(pos_ids, size=target_pos_count, replace=False).tolist()
    batch_neg_ids = np.random.choice(neg_ids, size=target_neg_count, replace=False).tolist()

    batch_ids = batch_pos_ids + batch_neg_ids
    np.random.shuffle(batch_ids)

    return batch_ids

def train_model_improved(model: SimpleGAT,
                        data_loader: DataLoader,
                        train_ids: List[str],
                        val_ids: List[str],
                        num_epochs: int = 15,
                        lr: float = 0.001,
                        batch_size: int = 16):
    """改进的训练函数"""
    
    # 使用AdamW优化器，更好的权重衰减
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, verbose=True
    )
    
    device = next(model.parameters()).device
    print(f"\n使用设备: {device}")

    # 计算类别权重
    print("\n计算类别权重...")
    pos_samples = 0
    neg_samples = 0
    
    for i in range(0, len(train_ids), batch_size):
        batch_ids = train_ids[i:i + batch_size]
        batch_data = data_loader.load_batch(batch_ids)
        if batch_data is None:
            continue
        
        batch_labels = batch_data['labels'].to(device)
        if batch_labels is not None:
            pos_samples += (batch_labels == 1).float().sum().item()
            neg_samples += (batch_labels == 0).float().sum().item()

    if pos_samples > 0 and neg_samples > 0:
        # 计算更强的类别权重
        imbalance_ratio = neg_samples / pos_samples
        print(f"类别不平衡比例: {imbalance_ratio:.2f}")
        
        # 使用更激进的权重策略
        pos_weight = min(imbalance_ratio * 1.5, 10.0)  # 限制最大权重
        neg_weight = 1.0
        
        class_weights = torch.tensor([neg_weight, pos_weight], device=device)
        print(f"类别权重 - 非霸凌: {neg_weight:.2f}, 霸凌: {pos_weight:.2f}")
        
        # 使用Focal Loss
        criterion = FocalLoss(alpha=pos_weight, gamma=2.0)
        print("使用Focal Loss")
    else:
        criterion = nn.CrossEntropyLoss()
        print("使用默认CrossEntropyLoss")

    best_val_f1 = 0.0
    patience = 8
    patience_counter = 0

    print("\n开始改进训练...")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        valid_batches = 0

        # 训练阶段 - 增加正样本比例
        num_batches = (len(train_ids) + batch_size - 1) // batch_size
        for _ in range(num_batches):
            batch_ids = create_balanced_batch(data_loader, train_ids, batch_size, target_pos_ratio=0.6)
            batch_data = data_loader.load_batch(batch_ids)

            if batch_data is None:
                continue

            valid_batches += 1

            try:
                x_dict = {k: v.to(device) for k, v in batch_data['x_dict'].items()}
                edge_index_dict = {k: v.to(device) for k, v in batch_data['edge_index_dict'].items()}
                if 'batch_dict' in batch_data:
                    edge_index_dict['batch_dict'] = {k: v.to(device) for k, v in batch_data['batch_dict'].items()}
                batch_labels = batch_data['labels'].to(device).view(-1).long()

                outputs = model(x_dict, edge_index_dict)
                loss = criterion(outputs, batch_labels)

                optimizer.zero_grad()
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                total_loss += loss.item()

            except RuntimeError as e:
                print(f"\nError in batch {valid_batches}: {str(e)}")
                continue

        avg_train_loss = total_loss / max(1, valid_batches)

        # 验证阶段
        model.eval()
        all_outputs = []
        all_labels = []

        with torch.no_grad():
            for i in range(0, len(val_ids), batch_size):
                batch_ids = val_ids[i:i + batch_size]
                batch_data = data_loader.load_batch(batch_ids)

                if batch_data is None:
                    continue

                try:
                    x_dict = {k: v.to(device) for k, v in batch_data['x_dict'].items()}
                    edge_index_dict = {k: v.to(device) for k, v in batch_data['edge_index_dict'].items()}
                    if 'batch_dict' in batch_data:
                        edge_index_dict['batch_dict'] = {k: v.to(device) for k, v in batch_data['batch_dict'].items()}
                    batch_labels = batch_data['labels'].to(device).view(-1).long()

                    outputs = model(x_dict, edge_index_dict)
                    all_outputs.append(outputs)
                    all_labels.append(batch_labels)

                except RuntimeError as e:
                    continue

        # 计算指标
        if len(all_outputs) > 0 and len(all_labels) > 0:
            all_outputs = torch.cat(all_outputs, dim=0)
            all_labels = torch.cat(all_labels, dim=0)
            metrics = calculate_metrics(all_outputs, all_labels)

            print(f"\n{'='*10} Epoch {epoch+1}/{num_epochs} {'='*10}")
            print(f"训练损失: {avg_train_loss:.4f}")
            print(f"准确率: {metrics['accuracy']:.4f}, AUC: {metrics.get('auc', 0.0):.4f}")
            print(f"霸凌 F1: {metrics['bullying']['f1']:.4f} (P: {metrics['bullying']['precision']:.4f}, R: {metrics['bullying']['recall']:.4f})")
            print(f"非霸凌 F1: {metrics['non_bullying']['f1']:.4f} (P: {metrics['non_bullying']['precision']:.4f}, R: {metrics['non_bullying']['recall']:.4f})")

            cm = metrics['confusion_matrix']
            print(f"混淆矩阵: TP={cm['true_positive']}, FP={cm['false_positive']}, TN={cm['true_negative']}, FN={cm['false_negative']}")

            # 学习率调度
            scheduler.step(metrics['bullying']['f1'])

            # 早停
            if metrics['bullying']['f1'] > best_val_f1:
                best_val_f1 = metrics['bullying']['f1']
                patience_counter = 0
                torch.save(model.state_dict(), "models/best_model_improved.pt")
                print("保存改进的最佳模型")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

    return model

if __name__ == "__main__":
    # 这里可以添加主函数来测试改进的训练
    pass
