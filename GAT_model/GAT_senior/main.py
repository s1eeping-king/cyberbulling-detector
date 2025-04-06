import torch
import random
from data_loader import DataLoader
from HGT_model import HGTransformer
from train import train_model

def main():
    # 设置随机种子
    random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.deterministic = True
    
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
    model = HGTransformer(
        hidden_dim=128,
        num_layers=5,
        num_heads=4,
        dropout=0.2
    ).to(device)
    
    print("\n模型参数统计:")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    
    # 创建数据加载器
    data_loader = DataLoader(
        uri=neo4j_info.get('uri', "bolt://localhost:7687"),
        user=neo4j_info.get('user', "neo4j"),
        password=neo4j_info.get('password', "password"),
        device=device,
        debug=True
    )
    
    try:
        # 获取所有媒体会话ID并划分训练集和验证集
        all_ids = data_loader.get_all_media_sessions()
        random.shuffle(all_ids)
        split = int(len(all_ids) * 0.8)
        train_ids = all_ids[:split]
        val_ids = all_ids[split:]
        
        print(f"\n数据集统计:")
        print(f"总样本数: {len(all_ids)}")
        print(f"训练集样本数: {len(train_ids)}")
        print(f"验证集样本数: {len(val_ids)}")
        
        # 训练模型
        model = train_model(
            model=model,
            data_loader=data_loader,
            train_ids=train_ids,
            val_ids=val_ids,
            num_epochs=30,
            lr=0.00025,
            batch_size=64
        )
        
    finally:
        # 确保关闭数据库连接
        data_loader.close()

if __name__ == "__main__":
    main()