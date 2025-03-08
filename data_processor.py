import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import json
import logging
from typing import Dict, List, Tuple, Optional
import os
import glob
from torch.utils.data import random_split
import torch.nn.functional as F
import pickle
from torch_geometric.data import Data as GraphData
from torch_geometric.loader import DataLoader as GraphDataLoader

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MultimodalDataset(Dataset):
    def __init__(self, vine_labeled_path, text_features_path, frame_features_dir, comment_ids_path, graph_path='knowledge_graph.pkl'):
        # 加载videolink到postid的映射
        self.url_to_postid = {}
        with open('urls_to_postids.txt', 'r') as f:
            for line in f:
                postid, url = line.strip().split(',')
                self.url_to_postid[url] = postid
        logger.info(f"Loaded {len(self.url_to_postid)} url-postid mappings")
        
        # 加载标注数据
        self.df = pd.read_csv(vine_labeled_path)
        
        # 加载所有评论特征
        self.text_features = np.load(text_features_path)
        logger.info(f"Loaded text features with shape: {self.text_features.shape}")
        
        # 加载评论ID映射并创建postid到特征索引的映射
        self.postid_to_feature_indices = {}
        with open(comment_ids_path, 'r') as f:
            for idx, line in enumerate(f):
                comment_id = line.strip()
                postid = comment_id.split('_')[0]  # 从postid_commentid中提取postid
                if postid not in self.postid_to_feature_indices:
                    self.postid_to_feature_indices[postid] = []
                self.postid_to_feature_indices[postid].append(idx)
        logger.info(f"Created mapping for {len(self.postid_to_feature_indices)} posts' comment features")
        
        # 获取所有视频特征文件
        frame_files = glob.glob(os.path.join(frame_features_dir, '*.npy'))
        self.feature_file_map = {os.path.splitext(os.path.basename(f))[0]: f for f in frame_files}
        logger.info(f"Found {len(frame_files)} frame feature files")
        
        # 加载知识图谱
        try:
            with open(graph_path, 'rb') as f:
                self.graph = pickle.load(f)
            logger.info(f"Loaded knowledge graph with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
            
            # 创建节点ID到索引的映射
            self.node_to_idx = {node: idx for idx, node in enumerate(self.graph.nodes())}
            
            # 创建节点特征矩阵
            self.node_features = self._create_node_features()
            logger.info(f"Created node features with shape: {self.node_features.shape}")
            
            # 创建边索引矩阵
            self.edge_index = self._create_edge_index()
            logger.info(f"Created edge index with shape: {self.edge_index.shape}")
            
        except Exception as e:
            logger.warning(f"Failed to load knowledge graph: {str(e)}")
            self.graph = None
            self.node_to_idx = None
            self.node_features = None
            self.edge_index = None
        
        # 过滤掉没有视频特征的样本
        valid_samples = []
        for idx, row in self.df.iterrows():
            videolink = row['videolink']
            if videolink in self.url_to_postid:
                postid = self.url_to_postid[videolink]
                # 检查是否同时有视频特征和评论特征
                if postid in self.feature_file_map and postid in self.postid_to_feature_indices:
                    valid_samples.append(idx)
        
        self.df = self.df.iloc[valid_samples]
        logger.info(f"Kept {len(valid_samples)} samples with both video and comment features")

    def _create_node_features(self) -> torch.Tensor:
        """创建节点特征矩阵"""
        if self.graph is None:
            return None
            
        # 为每种节点类型分配一个唯一的索引
        node_types = {
            'Video': 0,
            'VideoContent': 1,
            'CommentSection': 2,
            'Comment': 3,
            'User': 4,
            'AggressionLabel': 5,
            'BullyingLabel': 6,
            'FrameFeatures': 7,
            'BertFeatures': 8,
            'MultimodalFeatures': 9,
            'VideoEmotion': 10,
            'CommentEmotion': 11,
            'AggregateEmotion': 12
        }
        
        # 创建节点特征矩阵
        num_nodes = self.graph.number_of_nodes()
        num_types = len(node_types)
        x = torch.zeros((num_nodes, num_types))
        
        # 为每个节点分配one-hot编码
        for i, (node, data) in enumerate(self.graph.nodes(data=True)):
            node_type = data.get('node_type', 'Unknown')
            if node_type in node_types:
                type_idx = node_types[node_type]
                x[i, type_idx] = 1
            
        return x

    def _create_edge_index(self) -> torch.Tensor:
        """创建边索引矩阵"""
        if self.graph is None:
            return None
            
        # 收集所有边的源节点和目标节点
        edge_list = []
        for edge in self.graph.edges():
            source, target = edge
            edge_list.append([self.node_to_idx[source], self.node_to_idx[target]])
            
        # 转换为张量
        edge_index = torch.tensor(edge_list, dtype=torch.long).t()
        return edge_index

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        videolink = row['videolink']
        post_id = self.url_to_postid[videolink]
        
        # 获取标签
        aggression_label = 1 if row['question1'] == 'aggression' else 0
        bullying_label = 1 if row['question2'] == 'bullying' else 0
        
        # 获取该视频的所有评论特征并平均
        if post_id in self.postid_to_feature_indices:
            feature_indices = self.postid_to_feature_indices[post_id]
            # 去掉中间的维度1，然后对所有评论特征取平均
            text_features = self.text_features[feature_indices].squeeze(1).mean(axis=0)  # [768]
        else:
            logger.warning(f"No comment features found for post {post_id}")
            text_features = np.zeros(768)  # BERT特征维度
            
        # 加载视频特征
        video_feature_path = self.feature_file_map[post_id]
        video_features = np.load(video_feature_path)  # shape: [num_frames, 512]
        
        # 对视频特征在时间维度上进行平均池化
        video_features = torch.FloatTensor(video_features)  # 转换为tensor
        video_features = video_features.mean(dim=0)  # 对时间维度取平均，得到[512]维特征
        
        # 创建PyTorch Geometric数据对象
        data = GraphData(
            x=self.node_features,
            edge_index=self.edge_index,
            text_features=torch.FloatTensor(text_features).view(1, -1),  # [1, 768]
            video_features=video_features.view(1, -1),  # [1, 512]
            bullying_label=torch.LongTensor([bullying_label]),
            aggression_label=torch.LongTensor([aggression_label])
        )
        
        # 打印特征维度用于调试
        logger.debug(f"Text features shape: {text_features.shape}")
        logger.debug(f"Video features shape: {video_features.shape}")
        logger.debug(f"Graph data: {data}")
        
        return data

def create_data_loaders(
    vine_labeled_path: str,
    text_features_path: str,
    frame_features_dir: str,
    comment_ids_path: str,
    graph_path: str = 'knowledge_graph.pkl',
    batch_size: int = 32,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15
):
    """
    创建数据加载器
    
    Args:
        vine_labeled_path: 标注数据文件路径
        text_features_path: 评论特征文件路径
        frame_features_dir: 视频特征目录
        comment_ids_path: 评论ID文件路径
        graph_path: 知识图谱文件路径
        batch_size: 批次大小
        train_ratio: 训练集比例
        val_ratio: 验证集比例
    
    Returns:
        train_loader: 训练集数据加载器
        val_loader: 验证集数据加载器
        test_loader: 测试集数据加载器
    """
    # 创建数据集
    dataset = MultimodalDataset(
        vine_labeled_path=vine_labeled_path,
        text_features_path=text_features_path,
        frame_features_dir=frame_features_dir,
        comment_ids_path=comment_ids_path,
        graph_path=graph_path
    )
    
    # 计算数据集大小
    total_size = len(dataset)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    test_size = total_size - train_size - val_size
    
    # 分割数据集
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, 
        [train_size, val_size, test_size]
    )
    
    # 创建数据加载器
    train_loader = GraphDataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True
    )
    val_loader = GraphDataLoader(
        val_dataset, 
        batch_size=batch_size
    )
    test_loader = GraphDataLoader(
        test_dataset, 
        batch_size=batch_size
    )
    
    logger.info(f"Created data loaders with:")
    logger.info(f"- Training samples: {len(train_dataset)}")
    logger.info(f"- Validation samples: {len(val_dataset)}")
    logger.info(f"- Test samples: {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    # 测试数据加载器
    train_loader, val_loader, test_loader = create_data_loaders(
        vine_labeled_path="label_data/vine_labeled_cyberbullying_data.csv",
        text_features_path="processed/vine_features.npy",
        frame_features_dir="processed/frame_features",
        comment_ids_path="processed/comment_ids.txt",
        batch_size=32
    )
    
    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of validation batches: {len(val_loader)}")
    print(f"Number of test batches: {len(test_loader)}")
    
    # 测试一个批次
    for batch in train_loader:
        print("Batch shapes:")
        print(f"Text features: {batch['text_features'].shape}")
        print(f"Video features: {batch['video_features'].shape}")
        print(f"Aggression labels: {batch['aggression_label'].shape}")
        print(f"Bullying labels: {batch['bullying_label'].shape}")
        break 