import torch
import random
from data_loader import DataLoader
from GAT_v1 import HeteroGAT
from train import train_model

def main():
    # 设置随机种子
    random.seed(42)
    torch.manual_seed(42)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 加载Neo4j连接信息
    with open("knowledge_graph/neo4j_information.txt", 'r') as f:
        neo4j_info = {}
        for line in f:
            key, value = line.strip().split(' = ')
            neo4j_info[key] = value.strip('"')
    
    # 初始化模型和数据加载器
    model = HeteroGAT().to(device)
    data_loader = DataLoader(
        uri=neo4j_info['uri'],
        user=neo4j_info['username'],
        password=neo4j_info['password'],
        device=device
    )
    
    try:
        # 获取所有媒体会话ID并划分训练集和验证集
        all_ids = data_loader.get_all_media_sessions()
        random.shuffle(all_ids)  # 随机打乱数据
        split = int(len(all_ids) * 0.8)
        train_ids = all_ids[:split]
        val_ids = all_ids[split:]
        
        print(f"Total samples: {len(all_ids)}")
        print(f"Training samples: {len(train_ids)}")
        print(f"Validation samples: {len(val_ids)}")
        
        # 训练模型
        train_model(model, data_loader, train_ids, val_ids)
        
    finally:
        # 确保关闭数据库连接
        data_loader.close()

if __name__ == "__main__":
    main() 