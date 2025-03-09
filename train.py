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

def benchmark(model: CyberbullyingDetector, test_loader: DataLoader, save_dir: str = 'benchmark_results', device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
    """
    对模型进行全面的基准测试
    
    Args:
        model: 训练好的模型
        test_loader: 测试数据加载器
        save_dir: 保存结果的目录
        device: 计算设备
    """
    logger.info("开始基准测试...")
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    
    all_bullying_preds = []
    all_bullying_labels = []
    all_aggression_preds = []
    all_aggression_labels = []
    all_losses = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Benchmarking'):
            batch = batch.to(device)
            
            outputs = model(batch)
            loss, metrics = model.compute_loss(outputs, batch)
            
            all_losses.append(loss.item())
            all_bullying_preds.extend(outputs['bullying_logits'].argmax(dim=1).cpu().numpy())
            all_bullying_labels.extend(batch.bullying_label.cpu().numpy())
            all_aggression_preds.extend(outputs['aggression_logits'].argmax(dim=1).cpu().numpy())
            all_aggression_labels.extend(batch.aggression_label.cpu().numpy())
    
    # 计算并保存混淆矩阵
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    cm_bullying = confusion_matrix(all_bullying_labels, all_bullying_preds)
    sns.heatmap(cm_bullying, annot=True, fmt='d', cmap='Blues')
    plt.title('Cyberbullying Detection\nConfusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    
    plt.subplot(1, 2, 2)
    cm_aggression = confusion_matrix(all_aggression_labels, all_aggression_preds)
    sns.heatmap(cm_aggression, annot=True, fmt='d', cmap='Blues')
    plt.title('Aggression Detection\nConfusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confusion_matrices.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 生成分类报告
    bullying_report = classification_report(all_bullying_labels, all_bullying_preds)
    aggression_report = classification_report(all_aggression_labels, all_aggression_preds)
    
    # 计算每个类别的F1分数
    bullying_f1_per_class = f1_score(all_bullying_labels, all_bullying_preds, average=None)
    aggression_f1_per_class = f1_score(all_aggression_labels, all_aggression_preds, average=None)
    
    # 保存详细的评估结果
    results = {
        'average_loss': np.mean(all_losses),
        'bullying_f1_macro': f1_score(all_bullying_labels, all_bullying_preds, average='macro'),
        'aggression_f1_macro': f1_score(all_aggression_labels, all_aggression_preds, average='macro'),
        'bullying_f1_per_class': bullying_f1_per_class.tolist(),
        'aggression_f1_per_class': aggression_f1_per_class.tolist(),
        'bullying_confusion_matrix': cm_bullying.tolist(),
        'aggression_confusion_matrix': cm_aggression.tolist(),
        'bullying_classification_report': bullying_report,
        'aggression_classification_report': aggression_report
    }
    
    # 将结果写入文件
    with open(os.path.join(save_dir, 'benchmark_results.txt'), 'w', encoding='utf-8') as f:
        f.write("=== Cyberbullying Detection Model Benchmark Results ===\n\n")
        f.write(f"Average Loss: {results['average_loss']:.4f}\n\n")
        
        f.write("=== Cyberbullying Detection Results ===\n")
        f.write(f"Macro F1 Score: {results['bullying_f1_macro']:.4f}\n")
        f.write("F1 Score per class: \n")
        for i, f1 in enumerate(results['bullying_f1_per_class']):
            f.write(f"Class {i}: {f1:.4f}\n")
        f.write("\nClassification Report:\n")
        f.write(results['bullying_classification_report'])
        
        f.write("\n=== Aggression Detection Results ===\n")
        f.write(f"Macro F1 Score: {results['aggression_f1_macro']:.4f}\n")
        f.write("F1 Score per class: \n")
        for i, f1 in enumerate(results['aggression_f1_per_class']):
            f.write(f"Class {i}: {f1:.4f}\n")
        f.write("\nClassification Report:\n")
        f.write(results['aggression_classification_report'])
    
    # 保存结果为JSON格式（方便后续分析）
    with open(os.path.join(save_dir, 'benchmark_results.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    
    logger.info(f"基准测试结果已保存到 {save_dir}")
    return results

def main():
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 设置设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 创建数据加载器
    train_loader, val_loader, test_loader = create_data_loaders(
        vine_labeled_path="label_data/vine_labeled_cyberbullying_data.csv",
        text_features_path="processed/vine_features.npy",
        frame_features_dir="processed/frame_features",
        comment_ids_path="processed/comment_ids.txt",
        graph_path="knowledge_graph.pkl",
        batch_size=8
    )
    
    # 创建模型
    model = CyberbullyingDetector(
        text_dim=768,
        video_dim=512,
        hidden_dim=256,
        num_heads=2,
        dropout=0.3,
        node_feature_dim=13,
        gnn_hidden_dim=32,
        gnn_output_dim=64
    ).to(device)
    
    # 创建训练器
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        learning_rate=1e-4,
        weight_decay=1e-4,
        patience=5,
        gradient_accumulation_steps=4
    )
    
    # 训练模型
    trainer.train(num_epochs=20)
    
    # 进行基准测试
    logger.info("开始进行基准测试...")
    benchmark_dir = "benchmark_results"
    benchmark_results = benchmark(model, test_loader, save_dir=benchmark_dir, device=device)
    
    # 保存模型特征
    logger.info("保存模型特征...")
    features_dir = os.path.join(benchmark_dir, "model_features")
    save_features(model, train_loader, features_dir, split='train', device=device)
    save_features(model, val_loader, features_dir, split='val', device=device)
    save_features(model, test_loader, features_dir, split='test', device=device)

if __name__ == "__main__":
    main() 