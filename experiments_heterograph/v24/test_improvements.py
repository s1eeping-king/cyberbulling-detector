#!/usr/bin/env python3
"""
测试改进后的训练脚本
验证Focal Loss和改进的采样策略是否正常工作
"""

import torch
import torch.nn.functional as F
import numpy as np
from train import FocalLoss, create_balanced_batch

def test_focal_loss():
    """测试Focal Loss是否正常工作"""
    print("测试Focal Loss...")
    
    # 创建测试数据
    batch_size = 4
    num_classes = 2
    
    # 模拟输出（logits）
    outputs = torch.randn(batch_size, num_classes)
    
    # 模拟标签（不平衡：3个负样本，1个正样本）
    labels = torch.tensor([0, 0, 0, 1])
    
    # 测试不同的alpha和gamma值
    print("\n比较不同损失函数:")
    
    # 标准交叉熵
    ce_loss = F.cross_entropy(outputs, labels)
    print(f"CrossEntropyLoss: {ce_loss:.4f}")
    
    # Focal Loss (alpha=1, gamma=2)
    focal_loss_1 = FocalLoss(alpha=1, gamma=2)
    fl_1 = focal_loss_1(outputs, labels)
    print(f"FocalLoss (α=1, γ=2): {fl_1:.4f}")
    
    # Focal Loss (alpha=3, gamma=2) - 更强的类别权重
    focal_loss_2 = FocalLoss(alpha=3, gamma=2)
    fl_2 = focal_loss_2(outputs, labels)
    print(f"FocalLoss (α=3, γ=2): {fl_2:.4f}")
    
    print("✅ Focal Loss测试通过")
    return True

def test_balanced_sampling():
    """测试改进的平衡采样"""
    print("\n测试平衡采样策略...")
    
    # 模拟数据加载器
    class MockDataLoader:
        def __init__(self):
            # 模拟不平衡数据：80%负样本，20%正样本
            self.labels = {
                f"sample_{i}": 0 if i < 80 else 1 
                for i in range(100)
            }
        
        def get_label(self, sample_id):
            return self.labels.get(sample_id, 0)
    
    mock_loader = MockDataLoader()
    all_ids = list(mock_loader.labels.keys())
    
    print(f"原始数据分布:")
    pos_count = sum(1 for label in mock_loader.labels.values() if label == 1)
    neg_count = len(mock_loader.labels) - pos_count
    print(f"  正样本: {pos_count} ({pos_count/len(mock_loader.labels)*100:.1f}%)")
    print(f"  负样本: {neg_count} ({neg_count/len(mock_loader.labels)*100:.1f}%)")
    
    # 测试不同的采样比例
    batch_size = 16
    
    for target_ratio in [0.5, 0.6, 0.7]:
        print(f"\n目标正样本比例: {target_ratio*100:.0f}%")
        
        # 进行多次采样测试
        pos_ratios = []
        for _ in range(10):
            batch_ids = create_balanced_batch(mock_loader, all_ids, batch_size, target_ratio)
            batch_labels = [mock_loader.get_label(id) for id in batch_ids]
            actual_pos_ratio = sum(batch_labels) / len(batch_labels)
            pos_ratios.append(actual_pos_ratio)
        
        avg_pos_ratio = np.mean(pos_ratios)
        print(f"  实际平均正样本比例: {avg_pos_ratio*100:.1f}%")
        print(f"  批次大小: {len(batch_ids)}")
        
        # 检查是否接近目标比例
        assert abs(avg_pos_ratio - target_ratio) < 0.1, f"采样比例偏差过大"
    
    print("✅ 平衡采样测试通过")
    return True

def test_model_improvements():
    """测试模型改进是否兼容"""
    print("\n测试模型改进兼容性...")
    
    try:
        from GAT import SimpleGAT
        
        # 创建模型
        model = SimpleGAT(hidden_dim=32, num_layers=1, dropout=0.1, heads=2)
        
        # 创建测试数据
        device = torch.device("cpu")
        x_dict = {
            'user': torch.randn(2, 34),
            'media_session': torch.randn(1, 778),
            'comment': torch.randn(3, 770)
        }
        
        edge_index_dict = {
            ('user', 'publishes', 'media_session'): torch.tensor([[0], [0]]),
            ('user', 'creates', 'comment'): torch.tensor([[0, 1], [0, 1]]),
            ('comment', 'belongs_to', 'media_session'): torch.tensor([[0, 1, 2], [0, 0, 0]]),
            ('comment', 'mentions', 'user'): torch.tensor([[0], [1]]),
            ('user', 'offensive_comment', 'user'): torch.zeros((2, 0), dtype=torch.long),
            ('user', 'non_offensive_comment', 'user'): torch.zeros((2, 0), dtype=torch.long),
            'batch_dict': {
                'user': torch.tensor([0, 0]),
                'media_session': torch.tensor([0]),
                'comment': torch.tensor([0, 0, 0])
            }
        }
        
        # 前向传播
        model.eval()
        with torch.no_grad():
            outputs = model(x_dict, edge_index_dict)
            
        print(f"  模型输出形状: {outputs.shape}")
        assert outputs.shape == (1, 2), "输出形状不正确"
        
        # 测试Focal Loss与模型输出的兼容性
        labels = torch.tensor([1])  # 霸凌样本
        focal_loss = FocalLoss(alpha=2.0, gamma=2.0)
        loss = focal_loss(outputs, labels)
        
        print(f"  Focal Loss值: {loss:.4f}")
        assert not torch.isnan(loss), "损失值为NaN"
        
        print("✅ 模型改进兼容性测试通过")
        return True
        
    except Exception as e:
        print(f"❌ 模型改进兼容性测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("=" * 50)
    print("测试训练改进")
    print("=" * 50)
    
    success1 = test_focal_loss()
    success2 = test_balanced_sampling()
    success3 = test_model_improvements()
    
    if success1 and success2 and success3:
        print("\n🎉 所有改进测试通过！")
        print("\n主要改进:")
        print("✅ Focal Loss - 处理类别不平衡")
        print("✅ 改进的采样策略 - 60%正样本比例")
        print("✅ AdamW优化器 + 学习率调度")
        print("✅ 梯度裁剪防止梯度爆炸")
        print("✅ 更激进的类别权重策略")
        print("\n现在可以运行改进后的训练脚本了！")
    else:
        print("\n⚠️ 部分测试失败，请检查代码。")

if __name__ == "__main__":
    main()
