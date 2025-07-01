#!/usr/bin/env python3
"""
测试v24版本的GAT架构
验证多节点类型融合的图表征是否正常工作
"""

import torch
import torch.nn.functional as F
from GAT import SimpleGAT

def test_gat_architecture():
    """测试GAT架构的基本功能"""
    print("测试v24版本的GAT架构...")
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 创建模型
    model = SimpleGAT(
        hidden_dim=16,  # 使用较小的维度进行测试
        num_layers=2,
        dropout=0.5,
        heads=2
    ).to(device)
    
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters())}")
    
    # 创建测试数据
    batch_size = 2
    
    # 节点特征
    x_dict = {
        'user': torch.randn(4, 34, device=device),           # 4个用户节点
        'media_session': torch.randn(2, 778, device=device), # 2个媒体会话节点
        'comment': torch.randn(6, 770, device=device)        # 6个评论节点
    }
    
    # 边索引（创建一些简单的连接）
    edge_index_dict = {
        ('user', 'publishes', 'media_session'): torch.tensor([[0, 1], [0, 1]], device=device),
        ('user', 'creates', 'comment'): torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]], device=device),
        ('comment', 'belongs_to', 'media_session'): torch.tensor([[0, 1, 2, 3, 4, 5], [0, 0, 0, 1, 1, 1]], device=device),
        ('comment', 'mentions', 'user'): torch.tensor([[0, 1], [1, 2]], device=device),
        ('user', 'offensive_comment', 'user'): torch.tensor([[0], [1]], device=device),
        ('user', 'non_offensive_comment', 'user'): torch.tensor([[2], [3]], device=device),
        
        # 批次索引 - 关键部分
        'batch_dict': {
            'user': torch.tensor([0, 0, 1, 1], device=device),           # 前2个用户属于子图0，后2个属于子图1
            'media_session': torch.tensor([0, 1], device=device),        # 每个子图1个媒体会话
            'comment': torch.tensor([0, 0, 0, 1, 1, 1], device=device)  # 前3个评论属于子图0，后3个属于子图1
        }
    }
    
    print("\n输入数据形状:")
    for node_type, features in x_dict.items():
        print(f"  {node_type}: {features.shape}")
    
    print("\n批次索引:")
    for node_type, batch_idx in edge_index_dict['batch_dict'].items():
        print(f"  {node_type}: {batch_idx}")
    
    # 前向传播
    model.eval()
    with torch.no_grad():
        try:
            outputs = model(x_dict, edge_index_dict)
            print(f"\n模型输出形状: {outputs.shape}")
            print(f"输出值: {outputs}")
            
            # 检查输出是否合理
            assert outputs.shape == (batch_size, 2), f"期望输出形状为 ({batch_size}, 2)，实际为 {outputs.shape}"
            
            # 检查是否包含NaN或Inf
            assert not torch.isnan(outputs).any(), "输出包含NaN值"
            assert not torch.isinf(outputs).any(), "输出包含Inf值"
            
            # 计算softmax概率
            probs = F.softmax(outputs, dim=1)
            print(f"Softmax概率: {probs}")
            
            # 获取预测
            _, predictions = torch.max(probs, dim=1)
            print(f"预测类别: {predictions}")
            
            print("\n✅ 架构测试通过！")
            return True
            
        except Exception as e:
            print(f"\n❌ 架构测试失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

def test_edge_cases():
    """测试边界情况"""
    print("\n测试边界情况...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleGAT(hidden_dim=16, num_layers=1, dropout=0.0, heads=1).to(device)
    
    # 测试1: 空的节点类型
    print("测试1: 某些节点类型为空")
    x_dict = {
        'user': torch.randn(2, 34, device=device),
        'media_session': torch.randn(1, 778, device=device),
        'comment': torch.zeros(0, 770, device=device)  # 空的评论节点
    }
    
    edge_index_dict = {
        ('user', 'publishes', 'media_session'): torch.tensor([[0], [0]], device=device),
        ('user', 'creates', 'comment'): torch.zeros((2, 0), dtype=torch.long, device=device),
        ('comment', 'belongs_to', 'media_session'): torch.zeros((2, 0), dtype=torch.long, device=device),
        ('comment', 'mentions', 'user'): torch.zeros((2, 0), dtype=torch.long, device=device),
        ('user', 'offensive_comment', 'user'): torch.zeros((2, 0), dtype=torch.long, device=device),
        ('user', 'non_offensive_comment', 'user'): torch.zeros((2, 0), dtype=torch.long, device=device),
        'batch_dict': {
            'user': torch.tensor([0, 0], device=device),
            'media_session': torch.tensor([0], device=device),
            'comment': torch.zeros(0, dtype=torch.long, device=device)
        }
    }
    
    try:
        with torch.no_grad():
            outputs = model(x_dict, edge_index_dict)
            print(f"  输出形状: {outputs.shape}")
            assert outputs.shape[0] == 1, "批次大小应该为1"
            print("  ✅ 空节点测试通过")
    except Exception as e:
        print(f"  ❌ 空节点测试失败: {str(e)}")
        return False
    
    print("\n✅ 所有边界情况测试通过！")
    return True

if __name__ == "__main__":
    print("=" * 50)
    print("GAT v24 架构测试")
    print("=" * 50)
    
    success1 = test_gat_architecture()
    success2 = test_edge_cases()
    
    if success1 and success2:
        print("\n🎉 所有测试通过！v24架构工作正常。")
    else:
        print("\n⚠️  部分测试失败，请检查代码。")
