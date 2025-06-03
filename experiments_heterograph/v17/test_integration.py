#!/usr/bin/env python3
"""
测试VADER情感分数集成的脚本
"""

import torch
import sys
import os

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader import DataLoader
from GAT import SimpleGAT

def test_vader_integration():
    """测试VADER情感分数集成"""
    print("开始测试VADER情感分数集成...")
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    try:
        # 读取Neo4j连接信息
        neo4j_info_path = "knowledge_graph/neo4j_information.txt"
        if not os.path.exists(neo4j_info_path):
            print(f"错误: 找不到Neo4j配置文件: {neo4j_info_path}")
            return False
            
        with open(neo4j_info_path, 'r') as f:
            neo4j_info = {}
            for line in f:
                if ' = ' in line:
                    key, value = line.strip().split(' = ')
                    value = value.strip('"')
                    neo4j_info[key] = value
        
        print("Neo4j配置加载成功")
        
        # 创建数据加载器
        print("创建数据加载器...")
        data_loader = DataLoader(
            uri=neo4j_info['uri'],
            user=neo4j_info['username'],
            password=neo4j_info['password'],
            device=device,
            debug=True
        )
        
        print("数据加载器创建成功")
        
        # 测试加载一个子图
        print("测试加载子图...")
        if data_loader.train_sessions:
            test_session_id = data_loader.train_sessions[0]
            print(f"测试会话ID: {test_session_id}")
            
            # 加载子图
            subgraph = data_loader.load_subgraph(test_session_id)
            if subgraph is not None:
                print("子图加载成功!")
                print(f"节点类型: {subgraph.node_types}")
                print(f"边类型: {subgraph.edge_types}")
                
                # 检查特征维度
                for node_type in subgraph.node_types:
                    if hasattr(subgraph[node_type], 'x'):
                        features = subgraph[node_type].x
                        print(f"{node_type} 节点特征维度: {features.shape}")
                
                # 创建GAT模型
                print("创建GAT模型...")
                model = SimpleGAT(
                    hidden_dim=64,
                    num_layers=2,
                    dropout=0.5,
                    heads=4
                ).to(device)
                
                print("GAT模型创建成功")
                
                # 测试批量加载
                print("测试批量加载...")
                batch_ids = data_loader.train_sessions[:2]  # 取前2个会话
                batch_data = data_loader.load_batch(batch_ids)
                
                if batch_data is not None:
                    print("批量数据加载成功!")
                    print(f"批量大小: {batch_data['batch_size']}")
                    
                    # 准备输入数据
                    x_dict = {k: v.to(device) for k, v in batch_data['x_dict'].items()}
                    edge_index_dict = {k: v.to(device) for k, v in batch_data['edge_index_dict'].items()}
                    
                    # 添加批次索引
                    if 'batch_dict' in batch_data:
                        edge_index_dict['batch_dict'] = {k: v.to(device) for k, v in batch_data['batch_dict'].items()}
                    
                    print("输入数据准备完成")
                    
                    # 测试模型前向传播
                    print("测试模型前向传播...")
                    model.eval()
                    with torch.no_grad():
                        outputs = model(x_dict, edge_index_dict)
                        print(f"模型输出形状: {outputs.shape}")
                        print(f"模型输出: {outputs}")
                    
                    print("模型前向传播测试成功!")
                    
                else:
                    print("批量数据加载失败")
                    return False
                    
            else:
                print("子图加载失败")
                return False
        else:
            print("没有找到训练会话")
            return False
            
        # 关闭数据加载器
        data_loader.close()
        print("测试完成，所有功能正常!")
        return True
        
    except Exception as e:
        print(f"测试过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_vader_integration()
    if success:
        print("\n✅ VADER情感分数集成测试成功!")
    else:
        print("\n❌ VADER情感分数集成测试失败!")
        sys.exit(1)
