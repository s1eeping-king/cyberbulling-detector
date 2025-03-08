import networkx as nx
import pandas as pd
import numpy as np
import json
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional
import os
import glob
from collections import defaultdict
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KnowledgeGraphBuilder:
    def __init__(self):
        self.G = nx.DiGraph()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        self.node_types = {
            'Video': '#FF9999',
            'VideoContent': '#FFB366',
            'CommentSection': '#99FF99',
            'Comment': '#99CCFF',
            'User': '#FF99CC',
            'AggressionLabel': '#FF6666',
            'BullyingLabel': '#FF6666',
            'FrameFeatures': '#CC99FF',
            'BertFeatures': '#99FFCC',
            'MultimodalFeatures': '#FFCC99',
            'VideoEmotion': '#99FF99',
            'CommentEmotion': '#99FF99',
            'AggregateEmotion': '#99FF99'
        }
        
    def load_data(self):
        """加载所有必要的数据"""
        try:
            logger.info("Loading data...")
            
            # 添加节点类型追踪
            self.unknown_nodes = set()
            
            # 加载视频数据
            self.video_data = []
            with open('SampledASONAMPosts.json', 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        video = json.loads(line.strip())
                        self.video_data.append(video)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Error parsing video data line: {e}")
                        continue
            
            # 加载评论数据
            self.comment_data = []
            with open('sampled_post-comments_vine.json', 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        comment = json.loads(line.strip())
                        self.comment_data.append(comment)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Error parsing comment data line: {e}")
                        continue
            
            # 加载用户数据
            self.user_data = []
            with open('vine_users_data.json', 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        user = json.loads(line.strip())
                        self.user_data.append(user)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Error parsing user data line: {e}")
                        continue
            
            # 创建用户ID集合用于验证
            self.valid_user_ids = {user['_id'] for user in self.user_data}
            
            # 加载标签数据
            self.cyberbullying_data = pd.read_csv('label_data/vine_labeled_cyberbullying_data.csv')
            self.emotion_data = pd.read_csv('label_data/aggregate video emotion survey.csv')
            
            # 加载特征数据
            try:
                # 加载视频帧特征
                self.frame_features = {}
                frame_features_dir = 'processed/frame_features'
                if os.path.exists(frame_features_dir):
                    for npy_file in glob.glob(os.path.join(frame_features_dir, '*.npy')):
                        post_id = os.path.splitext(os.path.basename(npy_file))[0]
                        features = np.load(npy_file)
                        if self.device.type == 'cuda':
                            features = torch.from_numpy(features).to(self.device)
                        self.frame_features[post_id] = features
                    logger.info(f"Loaded {len(self.frame_features)} frame feature files")
                else:
                    logger.warning("Frame features directory not found")
                    self.frame_features = None
            except Exception as e:
                logger.warning(f"Error loading frame features: {e}")
                self.frame_features = None
                
            try:
                bert_features = np.load('processed/vine_features.npy')
                if self.device.type == 'cuda':
                    self.bert_features = torch.from_numpy(bert_features).to(self.device)
                else:
                    self.bert_features = bert_features
                logger.info("Loaded BERT features")
            except Exception as e:
                logger.warning(f"Error loading BERT features: {e}")
                self.bert_features = None
            
            # 加载评论ID映射
            self.comment_ids = []
            with open('processed/comment_ids.txt', 'r') as f:
                self.comment_ids = [line.strip() for line in f]
            
            logger.info(f"Successfully loaded data:")
            logger.info(f"- Videos: {len(self.video_data)}")
            logger.info(f"- Comments: {len(self.comment_data)}")
            logger.info(f"- Users: {len(self.user_data)}")
            logger.info(f"- Valid user IDs: {len(self.valid_user_ids)}")
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def _add_node(self, node_id: str, node_type: str, **attrs):
        """添加节点的辅助方法，用于追踪Unknown节点"""
        if node_type not in self.node_types:
            self.unknown_nodes.add((node_id, node_type))
            logger.warning(f"Adding node with unknown type: {node_type} for node {node_id}")
        self.G.add_node(node_id, node_type=node_type, **attrs)
    
    def build_video_entities(self):
        """构建视频相关实体"""
        for video in self.video_data:
            video_id = video['_id']
            
            # 添加视频节点
            self._add_node(f"video_{video_id}", 
                          node_type='Video',
                          description=video.get('description', ''),
                          url=video.get('url', ''),
                          likes_count=video.get('likes_count', 0),
                          comment_count=len([c for c in self.comment_data if c['postId'] == video_id]))
            
            # 添加视频内容节点
            self._add_node(f"video_content_{video_id}",
                          node_type='VideoContent',
                          content=video.get('content', ''))
            
            # 添加视频特征节点（如果存在）
            if self.frame_features is not None and video_id in self.frame_features:
                self._add_node(f"frame_features_{video_id}",
                              node_type='FrameFeatures',
                              features=self.frame_features[video_id])
                self.G.add_edge(f"video_{video_id}", f"frame_features_{video_id}",
                              relation='HAS_FRAME_FEATURES')
            
            # 添加关系
            self.G.add_edge(f"video_{video_id}", f"video_content_{video_id}",
                          relation='HAS_CONTENT')
    
    def build_comment_entities(self):
        """构建评论相关实体"""
        # 按视频ID组织评论
        video_comments = {}
        for comment in self.comment_data:
            video_id = comment['postId']
            if video_id not in video_comments:
                video_comments[video_id] = []
            video_comments[video_id].append(comment)
        
        # 构建评论节点和关系
        for video_id, comments in video_comments.items():
            # 添加评论区节点
            comment_section_id = f"comment_section_{video_id}"
            self._add_node(comment_section_id,
                          node_type='CommentSection',
                          comment_count=len(comments))
            
            # 添加视频-评论区关系
            self.G.add_edge(f"video_{video_id}", comment_section_id,
                          relation='HAS_COMMENTS')
            
            # 添加评论节点
            for comment in comments:
                comment_id = comment['_id']
                # 获取评论的BERT特征
                bert_feature = None
                comment_key = f"{video_id}_{comment_id}"
                if comment_key in self.comment_ids:
                    idx = self.comment_ids.index(comment_key)
                    if self.bert_features is not None:
                        bert_feature = self.bert_features[idx]
                
                self._add_node(f"comment_{comment_id}",
                              node_type='Comment',
                              content=comment.get('commentText', ''),
                              timestamp=comment.get('created', ''),
                              bert_features=bert_feature)
                
                # 添加评论区-评论关系
                self.G.add_edge(comment_section_id, f"comment_{comment_id}",
                              relation='CONTAINS')
                
                # 添加用户-评论关系
                user_id = comment.get('username')
                if user_id:
                    self.G.add_edge(f"user_{user_id}", f"comment_{comment_id}",
                                  relation='COMMENTED')
    
    def build_user_entities(self):
        """构建用户相关实体"""
        logger.info("Building user entities...")
        
        # 创建用户节点
        for user in self.user_data:
            user_id = user['_id']
            self._add_node(f"user_{user_id}",
                        node_type='User',
                        profile=user.get('profile', {}))
        
        # 检查评论中的用户
        comment_users = {comment.get('username') for comment in self.comment_data if comment.get('username')}
        unknown_users = comment_users - self.valid_user_ids
        if unknown_users:
            logger.warning(f"Found {len(unknown_users)} users in comments that are not in user data")
            logger.warning(f"Sample of unknown users: {list(unknown_users)[:5]}")
    
    def build_label_entities(self):
        """构建标签相关实体"""
        for _, row in self.cyberbullying_data.iterrows():
            video_id = row['videolink']
            
            # 添加攻击性标签
            aggression_label = row['question1']
            self._add_node(f"aggression_label_{video_id}",
                          node_type='AggressionLabel',
                          label=aggression_label,
                          severity_level=self._calculate_severity(aggression_label))
            self.G.add_edge(f"video_{video_id}", f"aggression_label_{video_id}",
                          relation='HAS_OVERALL_LABEL')
            
            # 添加霸凌标签
            bullying_label = row['question2']
            self._add_node(f"bullying_label_{video_id}",
                          node_type='BullyingLabel',
                          label=bullying_label,
                          severity_level=self._calculate_severity(bullying_label))
            self.G.add_edge(f"video_{video_id}", f"bullying_label_{video_id}",
                          relation='HAS_OVERALL_LABEL')
    
    def build_emotion_entities(self):
        """构建情感相关实体"""
        for _, row in self.emotion_data.iterrows():
            video_id = row['videolink']
            
            # 添加视频情感
            self._add_node(f"video_emotion_{video_id}",
                          node_type='VideoEmotion',
                          emotion=row.get('emotion', ''),
                          emotion_intensity=self._calculate_emotion_intensity(row.get('emotion', '')))
            self.G.add_edge(f"video_{video_id}", f"video_emotion_{video_id}",
                          relation='HAS_EMOTION')
            
            # 添加评论区情感
            comment_section_id = f"comment_section_{video_id}"
            if comment_section_id in self.G:
                self._add_node(f"comment_emotion_{video_id}",
                              node_type='CommentEmotion',
                              emotion=row.get('comment_emotion', ''),
                              emotion_intensity=self._calculate_emotion_intensity(row.get('comment_emotion', '')))
                self.G.add_edge(comment_section_id, f"comment_emotion_{video_id}",
                              relation='HAS_EMOTION')
    
    def build_multimodal_features(self):
        """构建多模态特征"""
        logger.info("Building multimodal features...")
        
        # 先收集所有需要处理的视频ID
        video_ids = []
        for node in self.G.nodes():
            if node.startswith('video_'):
                video_id = node.split('_')[1]
                video_ids.append(video_id)
        
        # 然后处理每个视频
        for video_id in video_ids:
            # 检查是否有视频特征和BERT特征
            if f"frame_features_{video_id}" in self.G and f"comment_section_{video_id}" in self.G:
                # 添加多模态特征节点
                self._add_node(f"multimodal_features_{video_id}",
                              node_type='MultimodalFeatures')
                
                # 添加关系
                self.G.add_edge(f"video_{video_id}", f"multimodal_features_{video_id}",
                              relation='HAS_MULTIMODAL_FEATURES')
    
    def _calculate_severity(self, label: str) -> float:
        """计算标签的严重程度"""
        severity_map = {
            'noneAgg': 0.0,
            'aggression': 0.5,
            'noneBll': 0.0,
            'bullying': 1.0
        }
        return severity_map.get(label, 0.0)
    
    def _calculate_emotion_intensity(self, emotion: str) -> float:
        """计算情感的强度"""
        intensity_map = {
            'neutral': 0.0,
            'joy': 0.8,
            'sad': 0.7,
            'love': 0.9,
            'surprise': 0.6,
            'fear': 0.8,
            'anger': 0.9
        }
        return intensity_map.get(emotion, 0.0)
    
    def build_user_interactions(self):
        """构建用户交互关系"""
        # 添加用户发布关系
        for video in self.video_data:
            user_id = video.get('username')
            if user_id:
                self.G.add_edge(f"user_{user_id}", f"video_{video['_id']}",
                              relation='POSTED')
        
        # 添加用户互动关系
        user_interactions = defaultdict(set)
        for comment in self.comment_data:
            user_id = comment.get('username')
            video_id = comment.get('postId')
            if user_id and video_id:
                # 获取视频发布者
                video_user = next((v.get('username') for v in self.video_data 
                                 if v['_id'] == video_id), None)
                if video_user:
                    user_interactions[user_id].add(video_user)
        
        # 添加用户之间的互动关系
        for user_id, interacted_users in user_interactions.items():
            for target_id in interacted_users:
                self.G.add_edge(f"user_{user_id}", f"user_{target_id}",
                              relation='INTERACTED_WITH')
        
        logger.info("Built user interaction relationships")

    def build(self):
        """构建完整的知识图谱"""
        try:
            logger.info("Loading data...")
            self.load_data()
            
            logger.info("Building video entities...")
            self.build_video_entities()
            
            logger.info("Building comment entities...")
            self.build_comment_entities()
            
            logger.info("Building user entities...")
            self.build_user_entities()
            
            logger.info("Building label entities...")
            self.build_label_entities()
            
            logger.info("Building emotion entities...")
            self.build_emotion_entities()
            
            logger.info("Building multimodal features...")
            self.build_multimodal_features()
            
            logger.info("Building user interactions...")
            self.build_user_interactions()
            
            # 检查Unknown节点
            if self.unknown_nodes:
                logger.warning(f"\nFound {len(self.unknown_nodes)} nodes with unknown types:")
                type_counts = {}
                for _, node_type in self.unknown_nodes:
                    type_counts[node_type] = type_counts.get(node_type, 0) + 1
                for node_type, count in type_counts.items():
                    logger.warning(f"- {node_type}: {count} nodes")
            
            logger.info("Knowledge graph built successfully")
            return self.G
            
        except Exception as e:
            logger.error(f"Error building knowledge graph: {str(e)}")
            raise
    
    def save_graph(self, path: str = 'knowledge_graph.pkl'):
        """保存知识图谱"""
        try:
            # 使用pickle保存
            import pickle
            with open(path, 'wb') as f:
                pickle.dump(self.G, f)
            logger.info(f"Knowledge graph saved to {path}")
            
            # 打印图谱统计信息
            logger.info(f"Graph statistics:")
            logger.info(f"- Number of nodes: {self.G.number_of_nodes()}")
            logger.info(f"- Number of edges: {self.G.number_of_edges()}")
            logger.info(f"- Node types: {set(nx.get_node_attributes(self.G, 'node_type').values())}")
            logger.info(f"- Edge types: {set(nx.get_edge_attributes(self.G, 'relation').values())}")
            
        except Exception as e:
            logger.error(f"Error saving knowledge graph: {str(e)}")
            raise

if __name__ == "__main__":
    # 创建知识图谱构建器
    kg_builder = KnowledgeGraphBuilder()
    
    # 构建知识图谱
    G = kg_builder.build()
    
    # 保存知识图谱
    kg_builder.save_graph() 