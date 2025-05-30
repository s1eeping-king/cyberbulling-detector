import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import List
from data_loader import DataLoader
from HGT import SimpleHGT, calculate_metrics

def create_balanced_batch(data_loader: DataLoader, ids: List[str], batch_size: int, target_pos_ratio: float = 0.5):
    """创建类别平衡的批次

    参数:
        data_loader: 数据加载器
        ids: 所有可用的ID列表
        batch_size: 批次大小
        target_pos_ratio: 目标正样本比例

    返回:
        batch_ids: 平衡后的批次ID列表
    """
    # 获取所有ID的标签
    pos_ids = []
    neg_ids = []

    for id in ids:
        label = data_loader.get_label(id)
        if label == 1:  # 霸凌样本
            pos_ids.append(id)
        else:  # 非霸凌样本
            neg_ids.append(id)

    # 计算需要的正样本数量
    target_pos_count = int(batch_size * target_pos_ratio)
    target_pos_count = min(target_pos_count, len(pos_ids))  # 确保不超过可用的正样本数量

    # 计算需要的负样本数量
    target_neg_count = batch_size - target_pos_count
    target_neg_count = min(target_neg_count, len(neg_ids))  # 确保不超过可用的负样本数量

    # 调整批次大小（如果样本不足）
    actual_batch_size = target_pos_count + target_neg_count

    # 随机选择正负样本
    batch_pos_ids = np.random.choice(pos_ids, size=target_pos_count, replace=False).tolist()
    batch_neg_ids = np.random.choice(neg_ids, size=target_neg_count, replace=False).tolist()

    # 合并并打乱
    batch_ids = batch_pos_ids + batch_neg_ids
    np.random.shuffle(batch_ids)

    return batch_ids



def train_model(model: SimpleHGT,
               data_loader: DataLoader,
               train_ids: List[str],
               val_ids: List[str],
               num_epochs: int = 10,
               lr: float = 0.001,
               batch_size: int = 32):
    """训练模型"""
    # 使用更小的学习率和更强的权重衰减，防止过拟合
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)

    # 获取设备
    device = next(model.parameters()).device
    print(f"\n使用设备: {device}")

    # 先计算类别权重
    print("\n计算类别权重...")
    pos_samples = 0
    neg_samples = 0
    valid_samples = 0

    # 正确计算正负样本数
    for i in range(0, len(train_ids), batch_size):
        batch_ids = train_ids[i:i + batch_size]
        batch_data = data_loader.load_batch(batch_ids)
        if batch_data is None:
            continue

        batch_labels = batch_data['labels'].to(device)
        if batch_labels is not None:
            pos_samples += (batch_labels == 1).float().sum().item()
            neg_samples += (batch_labels == 0).float().sum().item()
            valid_samples += batch_labels.size(0)

    if valid_samples > 0:
        print(f"\n数据统计 (总样本数: {valid_samples}):")
        print(f"霸凌 - 正样本: {pos_samples:.0f}, 负样本: {neg_samples:.0f}")

        # 确保不会出现除零错误
        pos_samples = max(pos_samples, 1.0)
        neg_samples = max(neg_samples, 1.0)

        # 计算类别权重 - 使用更强的平衡方法
        # 计算类别不平衡比例
        imbalance_ratio = neg_samples / pos_samples
        print(f"类别不平衡比例: {imbalance_ratio:.2f}")

        # 使用标准交叉熵损失，不使用类别权重
        criterion = nn.CrossEntropyLoss()
        print("使用标准交叉熵损失，不使用类别权重")
    else:
        criterion = nn.CrossEntropyLoss()
        print("使用默认权重")

    best_val_loss = float('inf')
    best_val_f1 = 0.0
    patience = 10  # 增加耐心值
    patience_counter = 0

    print("\n开始训练...")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        valid_batches = 0



        # 训练阶段
        # 使用标准的批次采样
        for i in range(0, len(train_ids), batch_size):
            batch_ids = train_ids[i:i + batch_size]
            batch_data = data_loader.load_batch(batch_ids)

            if batch_data is None:
                continue

            valid_batches += 1

            try:
                # 准备输入数据并移动到正确的设备
                x_dict = {k: v.to(device) for k, v in batch_data['x_dict'].items()}
                edge_index_dict = {k: v.to(device) for k, v in batch_data['edge_index_dict'].items()}
                # 添加批次索引到edge_index_dict
                if 'batch_dict' in batch_data:
                    edge_index_dict['batch_dict'] = {k: v.to(device) for k, v in batch_data['batch_dict'].items()}
                # 处理边属性
                edge_attr_dict = None
                if 'edge_attr_dict' in batch_data:
                    edge_attr_dict = {k: v.to(device) for k, v in batch_data['edge_attr_dict'].items()}
                batch_labels = batch_data['labels'].to(device)

                # 前向传播
                outputs = model(x_dict, edge_index_dict, edge_attr_dict)

                # 对于CrossEntropyLoss，标签应该是长整型，且是一维的
                # 将形状为 [batch_size, 1] 的标签转换为形状为 [batch_size] 的标签
                batch_labels = batch_labels.view(-1).long()

                # 打印原始输出（每10个批次打印一次）
                if valid_batches % 10 == 0:
                    # 获取原始logits
                    print(f"\n批次 {valid_batches} 的原始输出 (logits):")
                    print(outputs[:5])  # 只打印前5个样本的输出

                    # 获取softmax后的概率
                    probs = F.softmax(outputs, dim=1)
                    print(f"Softmax后的概率:")
                    print(probs[:5])  # 只打印前5个样本的概率

                    # 获取预测类别
                    _, preds = torch.max(probs, dim=1)
                    print(f"预测类别:")
                    print(preds[:5])  # 只打印前5个样本的预测

                    # 打印真实标签
                    print(f"真实标签:")
                    print(batch_labels[:5])  # 只打印前5个样本的标签

                # 计算损失
                loss = criterion(outputs, batch_labels)

                # 反向传播和优化
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            except RuntimeError as e:
                print(f"\nError in batch {valid_batches}: {str(e)}")
                print("跳过此批次...")
                continue

        avg_train_loss = total_loss / max(1, valid_batches)

        # 验证阶段
        model.eval()
        val_loss = 0
        val_batch_count = 0
        all_outputs = []
        all_labels = []

        with torch.no_grad():
            for i in range(0, len(val_ids), batch_size):
                batch_ids = val_ids[i:i + batch_size]
                batch_data = data_loader.load_batch(batch_ids)

                if batch_data is None:
                    continue

                try:
                    val_batch_count += 1

                    # 准备输入数据并移动到正确的设备
                    x_dict = {k: v.to(device) for k, v in batch_data['x_dict'].items()}
                    edge_index_dict = {k: v.to(device) for k, v in batch_data['edge_index_dict'].items()}
                    # 添加批次索引到edge_index_dict
                    if 'batch_dict' in batch_data:
                        edge_index_dict['batch_dict'] = {k: v.to(device) for k, v in batch_data['batch_dict'].items()}
                    # 处理边属性
                    edge_attr_dict = None
                    if 'edge_attr_dict' in batch_data:
                        edge_attr_dict = {k: v.to(device) for k, v in batch_data['edge_attr_dict'].items()}
                    batch_labels = batch_data['labels'].to(device)

                    # 前向传播
                    outputs = model(x_dict, edge_index_dict, edge_attr_dict)

                    # 对于CrossEntropyLoss，标签应该是长整型，且是一维的
                    # 将形状为 [batch_size, 1] 的标签转换为形状为 [batch_size] 的标签
                    batch_labels = batch_labels.view(-1).long()

                    # 计算损失
                    loss = criterion(outputs, batch_labels)
                    val_loss += loss.item()

                    all_outputs.append(outputs)
                    all_labels.append(batch_labels)

                except RuntimeError as e:
                    print(f"\nError in validation batch {val_batch_count}: {str(e)}")
                    print("跳过此批次...")
                    continue

        avg_val_loss = val_loss / max(1, val_batch_count)

        # 计算指标
        if len(all_outputs) > 0 and len(all_labels) > 0:
            all_outputs = torch.cat(all_outputs, dim=0)
            all_labels = torch.cat(all_labels, dim=0)
            metrics = calculate_metrics(all_outputs, all_labels)

            # 打印详细的评估指标
            print(f"\n{'='*20} Epoch {epoch+1}/{num_epochs} {'='*20}")
            print(f"训练损失: {avg_train_loss:.4f}, 验证损失: {avg_val_loss:.4f}")
            print(f"训练批次数: {valid_batches}, 验证批次数: {val_batch_count}")

            # 总体指标
            print(f"\n--- 总体指标 ---")
            print(f"准确率 (Accuracy): {metrics['accuracy']:.4f}")
            if 'auc' in metrics:
                print(f"AUC-ROC: {metrics['auc']:.4f}")

            # 霸凌类指标
            print(f"\n--- 霸凌类指标 ---")
            print(f"精确率 (Precision): {metrics['bullying']['precision']:.4f}")
            print(f"召回率 (Recall): {metrics['bullying']['recall']:.4f}")
            print(f"F1分数 (F1 Score): {metrics['bullying']['f1']:.4f}")

            # 非霸凌类指标
            print(f"\n--- 非霸凌类指标 ---")
            print(f"精确率 (Precision): {metrics['non_bullying']['precision']:.4f}")
            print(f"召回率 (Recall): {metrics['non_bullying']['recall']:.4f}")
            print(f"F1分数 (F1 Score): {metrics['non_bullying']['f1']:.4f}")

            # 混淆矩阵
            print(f"\n--- 混淆矩阵 ---")
            print(f"真正例 (TP): {metrics['confusion_matrix']['true_positive']}")
            print(f"假正例 (FP): {metrics['confusion_matrix']['false_positive']}")
            print(f"真负例 (TN): {metrics['confusion_matrix']['true_negative']}")
            print(f"假负例 (FN): {metrics['confusion_matrix']['false_negative']}")
        else:
            print(f"Epoch {epoch+1}/{num_epochs} - "
                  f"Train Loss: {avg_train_loss:.4f}, "
                  f"Val Loss: {avg_val_loss:.4f}")
            print("警告: 没有收集到有效的验证集输出和标签")

        # 早停 - 基于F1分数和验证损失
        model_improved = False

        # 检查验证损失是否改善
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model_improved = True
            # 保存基于损失的最佳模型
            torch.save(model.state_dict(), "models/best_model_loss.pt")
            print("保存基于损失的最佳模型")

        # 检查F1分数是否改善（如果有计算指标）
        if len(all_outputs) > 0 and len(all_labels) > 0 and metrics['bullying']['f1'] > best_val_f1:
            best_val_f1 = metrics['bullying']['f1']
            model_improved = True
            # 保存基于F1的最佳模型
            torch.save(model.state_dict(), "models/best_model_f1.pt")
            print("保存基于F1分数的最佳模型")

        # 更新耐心计数器
        if model_improved:
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    return model

def main():
    """主函数"""
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 加载Neo4j连接信息
    with open("knowledge_graph/neo4j_information.txt", 'r') as f:
        neo4j_info = {}
        for line in f:
            key, value = line.strip().split(' = ')
            # 移除引号
            value = value.strip('"')
            neo4j_info[key] = value

    # 初始化数据加载器
    data_loader = DataLoader(
        uri=neo4j_info['uri'],
        user=neo4j_info['username'],
        password=neo4j_info['password'],
        device=device
    )

    # 获取所有媒体会话ID
    all_ids = data_loader.get_all_media_sessions()
    print(f"总共找到 {len(all_ids)} 个媒体会话")

    # 随机打乱
    np.random.shuffle(all_ids)

    # 划分训练集和验证集
    train_ratio = 0.8
    train_size = int(len(all_ids) * train_ratio)
    train_ids = all_ids[:train_size]
    val_ids = all_ids[train_size:]

    print(f"训练集大小: {len(train_ids)}")
    print(f"验证集大小: {len(val_ids)}")

    # 创建HGT模型
    model = SimpleHGT(
        hidden_dim=48,  # 适中的隐藏维度
        num_layers=1,   # 减少层数
        dropout=0.4,    # 适中的dropout
        num_heads=4     # HGT特有参数：注意力头数量
    ).to(device)

    # 创建保存模型的目录
    import os
    os.makedirs("models", exist_ok=True)

    # 训练模型
    train_model(
        model=model,
        data_loader=data_loader,
        train_ids=train_ids,
        val_ids=val_ids,
        num_epochs=50,   # 减少训练轮数，避免过拟合
        lr=0.0001,       # 适度的学习率
        batch_size=32    # 批次大小
    )

    # 关闭数据加载器
    data_loader.close()

    print("训练完成!")

if __name__ == "__main__":
    main()
