import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import precision_recall_curve, f1_score, precision_score, recall_score
from typing import Tuple, Dict
import matplotlib.pyplot as plt

def find_optimal_threshold(model, data_loader, val_ids: list, device, batch_size: int = 16) -> Tuple[float, Dict]:
    """找到最优的分类阈值来最大化F1分数"""
    
    model.eval()
    all_probs = []
    all_labels = []
    
    print("收集验证集预测概率...")
    
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
                probs = F.softmax(outputs, dim=1)
                
                # 获取霸凌类的概率
                bullying_probs = probs[:, 1].cpu().numpy()
                labels = batch_labels.cpu().numpy()
                
                all_probs.extend(bullying_probs)
                all_labels.extend(labels)
                
            except Exception as e:
                print(f"处理批次时出错: {e}")
                continue
    
    if len(all_probs) == 0:
        print("警告: 没有收集到有效的预测结果")
        return 0.5, {}
    
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    print(f"收集到 {len(all_probs)} 个样本")
    print(f"正样本数量: {np.sum(all_labels)}")
    print(f"负样本数量: {len(all_labels) - np.sum(all_labels)}")
    
    # 计算不同阈值下的指标
    thresholds = np.arange(0.1, 0.9, 0.01)
    best_threshold = 0.5
    best_f1 = 0.0
    results = []
    
    for threshold in thresholds:
        predictions = (all_probs >= threshold).astype(int)
        
        if len(np.unique(predictions)) < 2:
            # 如果所有预测都是同一类，跳过
            continue
            
        f1 = f1_score(all_labels, predictions)
        precision = precision_score(all_labels, predictions, zero_division=0)
        recall = recall_score(all_labels, predictions, zero_division=0)
        
        results.append({
            'threshold': threshold,
            'f1': f1,
            'precision': precision,
            'recall': recall
        })
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    print(f"\n最优阈值: {best_threshold:.3f}")
    print(f"最优F1分数: {best_f1:.4f}")
    
    # 计算最优阈值下的详细指标
    best_predictions = (all_probs >= best_threshold).astype(int)
    best_precision = precision_score(all_labels, best_predictions, zero_division=0)
    best_recall = recall_score(all_labels, best_predictions, zero_division=0)
    
    best_metrics = {
        'threshold': best_threshold,
        'f1': best_f1,
        'precision': best_precision,
        'recall': best_recall,
        'all_results': results
    }
    
    print(f"最优阈值下的指标:")
    print(f"  精确率: {best_precision:.4f}")
    print(f"  召回率: {best_recall:.4f}")
    print(f"  F1分数: {best_f1:.4f}")
    
    return best_threshold, best_metrics

def evaluate_with_threshold(model, data_loader, test_ids: list, device, threshold: float = 0.5, batch_size: int = 16) -> Dict:
    """使用指定阈值评估模型"""
    
    model.eval()
    all_probs = []
    all_labels = []
    
    print(f"使用阈值 {threshold:.3f} 评估模型...")
    
    with torch.no_grad():
        for i in range(0, len(test_ids), batch_size):
            batch_ids = test_ids[i:i + batch_size]
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
                probs = F.softmax(outputs, dim=1)
                
                bullying_probs = probs[:, 1].cpu().numpy()
                labels = batch_labels.cpu().numpy()
                
                all_probs.extend(bullying_probs)
                all_labels.extend(labels)
                
            except Exception as e:
                continue
    
    if len(all_probs) == 0:
        return {}
    
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    # 使用指定阈值进行预测
    predictions = (all_probs >= threshold).astype(int)
    
    # 计算指标
    f1 = f1_score(all_labels, predictions)
    precision = precision_score(all_labels, predictions, zero_division=0)
    recall = recall_score(all_labels, predictions, zero_division=0)
    accuracy = np.mean(predictions == all_labels)
    
    # 计算混淆矩阵
    tp = np.sum((predictions == 1) & (all_labels == 1))
    fp = np.sum((predictions == 1) & (all_labels == 0))
    tn = np.sum((predictions == 0) & (all_labels == 0))
    fn = np.sum((predictions == 0) & (all_labels == 1))
    
    # 计算非霸凌类的指标
    non_bullying_precision = tn / (tn + fn) if (tn + fn) > 0 else 0
    non_bullying_recall = tn / (tn + fp) if (tn + fp) > 0 else 0
    non_bullying_f1 = 2 * (non_bullying_precision * non_bullying_recall) / (non_bullying_precision + non_bullying_recall) if (non_bullying_precision + non_bullying_recall) > 0 else 0
    
    metrics = {
        'threshold': threshold,
        'accuracy': accuracy,
        'bullying': {
            'precision': precision,
            'recall': recall,
            'f1': f1
        },
        'non_bullying': {
            'precision': non_bullying_precision,
            'recall': non_bullying_recall,
            'f1': non_bullying_f1
        },
        'confusion_matrix': {
            'true_positive': int(tp),
            'false_positive': int(fp),
            'true_negative': int(tn),
            'false_negative': int(fn)
        }
    }
    
    print(f"\n使用阈值 {threshold:.3f} 的评估结果:")
    print(f"准确率: {accuracy:.4f}")
    print(f"霸凌 F1: {f1:.4f} (P: {precision:.4f}, R: {recall:.4f})")
    print(f"非霸凌 F1: {non_bullying_f1:.4f} (P: {non_bullying_precision:.4f}, R: {non_bullying_recall:.4f})")
    print(f"混淆矩阵: TP={tp}, FP={fp}, TN={tn}, FN={fn}")
    
    return metrics

def plot_threshold_analysis(results: list, save_path: str = "threshold_analysis.png"):
    """绘制阈值分析图"""
    
    if not results:
        print("没有结果可以绘制")
        return
    
    thresholds = [r['threshold'] for r in results]
    f1_scores = [r['f1'] for r in results]
    precisions = [r['precision'] for r in results]
    recalls = [r['recall'] for r in results]
    
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(thresholds, f1_scores, 'b-', linewidth=2)
    plt.xlabel('Threshold')
    plt.ylabel('F1 Score')
    plt.title('F1 Score vs Threshold')
    plt.grid(True)
    
    plt.subplot(2, 2, 2)
    plt.plot(thresholds, precisions, 'r-', linewidth=2, label='Precision')
    plt.plot(thresholds, recalls, 'g-', linewidth=2, label='Recall')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title('Precision and Recall vs Threshold')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 2, 3)
    plt.plot(precisions, recalls, 'purple', linewidth=2)
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    plt.title('Precision-Recall Curve')
    plt.grid(True)
    
    plt.subplot(2, 2, 4)
    # 找到最优F1分数
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]
    
    plt.plot(thresholds, f1_scores, 'b-', linewidth=2)
    plt.axvline(x=best_threshold, color='red', linestyle='--', linewidth=2)
    plt.axhline(y=best_f1, color='red', linestyle='--', linewidth=2)
    plt.xlabel('Threshold')
    plt.ylabel('F1 Score')
    plt.title(f'Best Threshold: {best_threshold:.3f} (F1: {best_f1:.4f})')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"阈值分析图已保存到: {save_path}")

if __name__ == "__main__":
    # 这里可以添加测试代码
    pass
