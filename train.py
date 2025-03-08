import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import logging
import json
from tqdm import tqdm
import numpy as np
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, f1_score
import seaborn as sns
import os

from data_processor import create_data_loaders
from model import CyberbullyingDetector
from torch_geometric.loader import DataLoader as GraphDataLoader

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def save_features(model: CyberbullyingDetector, 
                 data_loader: GraphDataLoader, 
                 save_dir: str,
                 split: str = 'train',
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
    """
    保存模型各阶段的特征
    
    Args:
        model: 模型
        data_loader: 数据加载器
        save_dir: 保存目录
        split: 数据集划分名称 ('train', 'val', 'test')
        device: 计算设备
    """
    logger.info(f"开始保存{split}集特征...")
    model.eval()
    features_dir = os.path.join(save_dir, 'features', split)
    os.makedirs(features_dir, exist_ok=True)
    
    all_features = {
        'text_features': [],
        'video_features': [],
        'text_encoded': [],
        'video_encoded': [],
        'text_attended': [],
        'fused_features': [],
        'gnn_features': [],
        'final_features': [],
        'labels': []
    }
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc=f"保存{split}集特征"):
            # 将数据移到设备上
            batch = batch.to(device)
            
            # 获取模型输出
            outputs = model(batch)
            
            # 收集特征和标签
            for key in all_features.keys():
                if key == 'labels':
                    all_features[key].append({
                        'bullying': batch.bullying_label.cpu().numpy(),
                        'aggression': batch.aggression_label.cpu().numpy()
                    })
                elif key in outputs and outputs[key] is not None:
                    all_features[key].append(outputs[key].cpu().numpy())
    
    # 合并并保存特征
    for key in all_features:
        if key == 'labels':
            labels = {
                'bullying': np.concatenate([x['bullying'] for x in all_features[key]]),
                'aggression': np.concatenate([x['aggression'] for x in all_features[key]])
            }
            np.savez(
                os.path.join(features_dir, f'{key}.npz'),
                bullying=labels['bullying'],
                aggression=labels['aggression']
            )
        elif all_features[key]:  # 只保存非空特征
            features = np.concatenate(all_features[key], axis=0)
            np.save(os.path.join(features_dir, f'{key}.npy'), features)
    
    logger.info(f"特征已保存到 {features_dir}")

class Trainer:
    def __init__(
        self,
        model: CyberbullyingDetector,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        patience: int = 5,
        gradient_accumulation_steps: int = 4  # 添加梯度累积步数
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.patience = patience
        self.gradient_accumulation_steps = gradient_accumulation_steps
        
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=3,
            verbose=True
        )
        
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_bullying_acc': [],
            'train_aggression_acc': [],
            'val_bullying_acc': [],
            'val_aggression_acc': [],
            'train_bullying_f1': [],
            'train_aggression_f1': [],
            'val_bullying_f1': [],
            'val_aggression_f1': []
        }
        
    def train_epoch(self) -> Dict[str, float]:
        self.model.train()
        total_loss = 0
        all_metrics = []
        
        all_bullying_preds = []
        all_bullying_labels = []
        all_aggression_preds = []
        all_aggression_labels = []
        
        self.optimizer.zero_grad()  # 在epoch开始时清零梯度
        
        for i, batch in enumerate(tqdm(self.train_loader, desc='Training')):
            # 将数据移到设备上
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}
            
            # 前向传播
            outputs = self.model(batch)
            loss, metrics = self.model.compute_loss(outputs, batch)
            
            # 缩放损失以适应梯度累积
            loss = loss / self.gradient_accumulation_steps
            
            # 反向传播
            loss.backward()
            
            # 梯度累积
            if (i + 1) % self.gradient_accumulation_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
            
            total_loss += loss.item() * self.gradient_accumulation_steps
            all_metrics.append(metrics)
            
            # 收集预测和标签
            all_bullying_preds.extend(outputs['bullying_logits'].argmax(dim=1).cpu().numpy())
            all_bullying_labels.extend(batch['bullying_label'].squeeze().cpu().numpy())
            all_aggression_preds.extend(outputs['aggression_logits'].argmax(dim=1).cpu().numpy())
            all_aggression_labels.extend(batch['aggression_label'].squeeze().cpu().numpy())
            
            # 清理GPU缓存
            if i % 10 == 0:
                torch.cuda.empty_cache()
        
        # 处理最后一个不完整的梯度累积
        if (i + 1) % self.gradient_accumulation_steps != 0:
            self.optimizer.step()
            self.optimizer.zero_grad()
        
        # 计算F1分数
        bullying_f1 = f1_score(all_bullying_labels, all_bullying_preds, average='macro')
        aggression_f1 = f1_score(all_aggression_labels, all_aggression_preds, average='macro')
        
        # 计算平均指标
        avg_metrics = {
            k: np.mean([m[k] for m in all_metrics])
            for k in all_metrics[0].keys()
        }
        avg_metrics['loss'] = total_loss / len(self.train_loader)
        avg_metrics['bullying_f1'] = bullying_f1
        avg_metrics['aggression_f1'] = aggression_f1
        
        return avg_metrics
    
    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0
        all_metrics = []
        
        all_bullying_preds = []
        all_bullying_labels = []
        all_aggression_preds = []
        all_aggression_labels = []
        
        for batch in tqdm(loader, desc='Evaluating'):
            # 将数据移到设备上
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}
            
            # 前向传播
            outputs = self.model(batch)
            loss, metrics = self.model.compute_loss(outputs, batch)
            
            total_loss += loss.item()
            all_metrics.append(metrics)
            
            # 收集预测和标签
            all_bullying_preds.extend(outputs['bullying_logits'].argmax(dim=1).cpu().numpy())
            all_bullying_labels.extend(batch['bullying_label'].squeeze().cpu().numpy())
            all_aggression_preds.extend(outputs['aggression_logits'].argmax(dim=1).cpu().numpy())
            all_aggression_labels.extend(batch['aggression_label'].squeeze().cpu().numpy())
        
        # 计算F1分数
        bullying_f1 = f1_score(all_bullying_labels, all_bullying_preds, average='macro')
        aggression_f1 = f1_score(all_aggression_labels, all_aggression_preds, average='macro')
        
        # 计算平均指标
        avg_metrics = {
            k: np.mean([m[k] for m in all_metrics])
            for k in all_metrics[0].keys()
        }
        avg_metrics['loss'] = total_loss / len(loader)
        avg_metrics['bullying_f1'] = bullying_f1
        avg_metrics['aggression_f1'] = aggression_f1
        
        return avg_metrics
    
    def train(self, num_epochs: int):
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(num_epochs):
            logger.info(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            # Training
            train_metrics = self.train_epoch()
            logger.info(
                f"Train - Loss: {train_metrics['loss']:.4f}, "
                f"Bullying Acc: {train_metrics['bullying_acc']:.4f}, "
                f"Aggression Acc: {train_metrics['aggression_acc']:.4f}, "
                f"Bullying F1: {train_metrics['bullying_f1']:.4f}, "
                f"Aggression F1: {train_metrics['aggression_f1']:.4f}"
            )
            
            # Validation
            val_metrics = self.evaluate(self.val_loader)
            logger.info(
                f"Val - Loss: {val_metrics['loss']:.4f}, "
                f"Bullying Acc: {val_metrics['bullying_acc']:.4f}, "
                f"Aggression Acc: {val_metrics['aggression_acc']:.4f}, "
                f"Bullying F1: {val_metrics['bullying_f1']:.4f}, "
                f"Aggression F1: {val_metrics['aggression_f1']:.4f}"
            )
            
            # Update learning rate
            self.scheduler.step(val_metrics['loss'])
            
            # Early stopping check
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= self.patience:
                logger.info(f"Early stopping triggered after {epoch + 1} epochs!")
                break
        
        # Evaluate on test set
        logger.info("\nEvaluating on test set...")
        test_metrics = self.evaluate(self.test_loader)
        logger.info(
            f"Test - Loss: {test_metrics['loss']:.4f}, "
            f"Bullying Acc: {test_metrics['bullying_acc']:.4f}, "
            f"Aggression Acc: {test_metrics['aggression_acc']:.4f}, "
            f"Bullying F1: {test_metrics['bullying_f1']:.4f}, "
            f"Aggression F1: {test_metrics['aggression_f1']:.4f}"
        )

def main():
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 创建数据加载器
    train_loader, val_loader, test_loader = create_data_loaders(
        vine_labeled_path="label_data/vine_labeled_cyberbullying_data.csv",
        text_features_path="processed/vine_features.npy",
        frame_features_dir="processed/frame_features",
        comment_ids_path="processed/comment_ids.txt",
        graph_path="knowledge_graph.pkl",
        batch_size=8  # 减小batch size
    )
    
    # 创建模型
    model = CyberbullyingDetector(
        text_dim=768,
        video_dim=512,
        hidden_dim=256,  # 减小隐藏层维度
        num_heads=2,     # 减少注意力头数量
        dropout=0.3,
        node_feature_dim=13,  # 更新为实际的节点类型数量
        gnn_hidden_dim=32,   # 减小GNN隐藏层维度
        gnn_output_dim=64    # 减小GNN输出维度
    )
    
    # 创建训练器
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        learning_rate=1e-4,
        weight_decay=1e-4,
        patience=5,
        gradient_accumulation_steps=4  # 添加梯度累积
    )
    
    # 训练模型
    trainer.train(num_epochs=20)

if __name__ == "__main__":
    main() 