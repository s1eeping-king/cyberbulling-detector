import networkx as nx
import pandas as pd
import numpy as np
import json
import logging
import os
import glob
import torch
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict
from pathlib import Path
from bc_fusion import MultimodalFusion
from data_loader import DataLoader, load_all_data

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CyberbullyingKnowledgeGraph:
    """
    霸凌检测知识图谱构建器
    基于视频、用户、媒体会话和标签的异构图结构
    """
    def __init__(self, data_loader=None):
        # 初始化图结构
        self.G = nx.DiGraph()
        
        # 设置设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # 定义节点类型
        self.node_types = {
            'Video',           # 视频节点
            'User',            # 用户节点
            'MediaSession',    # 媒体会话节点
            'Label'            # 标签节点
        }
        
        # 定义关系类型
        self.relation_types = {
            'USER_POSTED_VIDEO',    # 用户发布视频
            'USER_COMMENTED_VIDEO', # 用户评论视频
            'VIDEO_HAS_SESSION',    # 视频拥有媒体会话
            'HAS_LABEL'             # 实体拥有标签
        }
        
        # 数据加载器
        self.data_loader = data_loader if data_loader is not None else DataLoader(self.device)
        
        # 统计信息
        self.stats = {
            'video_count': 0,
            'user_count': 0,
            'media_session_count': 0,
            'label_count': 0,
            'relation_count': 0
        }
        
        # 特征融合器
        self.fusion_model = None
        
    def load_data(self):
        """加载所有必要的数据"""
        return self.data_loader.load_data()

    def calculate_user_activity(self) -> Dict[str, Dict]:
        """计算用户活跃度"""
        user_activity = defaultdict(lambda: {'post_count': 0, 'comment_count': 0, 'total_interactions': 0})
        
        # 统计发布视频数
        for video in self.data_loader.video_data:
            user_id = video.get('username')
            if user_id:
                user_activity[user_id]['post_count'] += 1
                user_activity[user_id]['total_interactions'] += 1
        
        # 统计评论数
        for comment in self.data_loader.comment_data:
            user_id = comment.get('username')
            if user_id:
                user_activity[user_id]['comment_count'] += 1
                user_activity[user_id]['total_interactions'] += 1
        
        # 计算活跃度分数 (简单加权)
        for user_id, stats in user_activity.items():
            stats['activity_score'] = stats['post_count'] * 2 + stats['comment_count']
        
        return user_activity

    def get_active_users(self, min_interactions: int = 3) -> Set[str]:
        """获取活跃用户ID集合"""
        user_activity = self.calculate_user_activity()
        return {user_id for user_id, stats in user_activity.items() 
                if stats['total_interactions'] >= min_interactions}

    def get_post_id_from_url(self, videolink: str) -> Optional[str]:
        """从视频链接获取postId"""
        return self.data_loader.get_post_id_from_url(videolink)

    def get_comments_for_video(self, video_id: str) -> List[Dict]:
        """获取视频的所有评论"""
        return self.data_loader.get_comments_for_video(video_id)

    def get_comment_features(self, video_id: str) -> Optional[np.ndarray]:
        """获取视频评论的特征向量"""
        return self.data_loader.get_comment_features(video_id)

    def build_video_nodes(self):
        """构建视频节点"""
        logger.info("Building video nodes...")
        for video in self.data_loader.video_data:
            video_id = video['_id']
            
            # 添加视频节点
            self.G.add_node(
                f"video_{video_id}",
                node_type='Video',
                id=video_id,
                url=video.get('url', ''),
                description=video.get('description', ''),
                likes_count=video.get('likes_count', 0),
                timestamp=video.get('created', ''),
                comment_count=len(self.get_comments_for_video(video_id))
            )
            
            # 添加视频帧特征
            if video_id in self.data_loader.frame_features:
                self.G.nodes[f"video_{video_id}"]["frame_features"] = self.data_loader.frame_features[video_id]
            
            self.stats['video_count'] += 1
        
        logger.info(f"Built {self.stats['video_count']} video nodes")

    def build_user_nodes(self, min_interactions: int = 3):
        """构建用户节点"""
        logger.info("Building user nodes...")
        
        # 获取活跃用户
        active_users = self.get_active_users(min_interactions)
        user_activity = self.calculate_user_activity()
        
        # 如果没有活跃用户，降低阈值
        if len(active_users) == 0:
            logger.warning(f"No active users found with threshold {min_interactions}, lowering to 1")
            active_users = self.get_active_users(1)
        
        # 构建用户节点
        for user_id in active_users:
            # 获取用户资料
            user_profile = next((user.get('profile', {}) for user in self.data_loader.user_data if user['_id'] == user_id), {})
            
            # 添加用户节点
            self.G.add_node(
                f"user_{user_id}",
                node_type='User',
                id=user_id,
                profile=user_profile,
                post_count=user_activity[user_id]['post_count'],
                comment_count=user_activity[user_id]['comment_count'],
                activity_score=user_activity[user_id]['activity_score']
            )
            
            self.stats['user_count'] += 1
        
        logger.info(f"Built {self.stats['user_count']} user nodes (from {len(self.data_loader.valid_user_ids)} total users)")

    def build_media_session_nodes(self):
        """构建媒体会话节点"""
        logger.info("Building media session nodes...")
        
        # 获取所有视频ID
        video_ids = [node.split('_')[1] for node in self.G.nodes() if node.startswith('video_')]
        
        for video_id in video_ids:
            # 获取视频的所有评论
            comments = self.get_comments_for_video(video_id)
            
            # 聚合评论内容
            aggregated_comments = " ".join([comment.get('commentText', '') for comment in comments])
            
            # 获取评论特征
            comment_features = self.get_comment_features(video_id)
            
            # 添加媒体会话节点
            self.G.add_node(
                f"media_session_{video_id}",
                node_type='MediaSession',
                video_id=video_id,
                comment_count=len(comments),
                aggregated_comments=aggregated_comments
            )
            
            # 添加评论特征
            if comment_features is not None:
                self.G.nodes[f"media_session_{video_id}"]["session_features"] = torch.from_numpy(comment_features).float()
            
            # 添加视频-媒体会话关系
            self.G.add_edge(
                f"video_{video_id}",
                f"media_session_{video_id}",
                relation='VIDEO_HAS_SESSION',
                creation_time=self.G.nodes[f"video_{video_id}"].get('timestamp', '')
            )
            
            self.stats['media_session_count'] += 1
            self.stats['relation_count'] += 1
        
        logger.info(f"Built {self.stats['media_session_count']} media session nodes")

    def build_label_nodes(self):
        """构建标签节点"""
        logger.info("Building label nodes...")
        
        # 构建霸凌和攻击性标签
        label_count = 0
        for _, row in self.data_loader.cyberbullying_data.iterrows():
            videolink = row['videolink']
            video_id = self.get_post_id_from_url(videolink)
            
            # 如果找不到映射，尝试直接匹配视频URL
            if not video_id:
                for v in self.data_loader.video_data:
                    post_id = v.get('_id')
                    url = v.get('url', '')
                    
                    if url and (videolink in url or url.endswith(videolink)):
                        video_id = post_id
                        # 更新映射
                        self.data_loader.videolink_to_postid[videolink] = video_id
                        self.data_loader.postid_to_videolink[video_id] = videolink
                        logger.info(f"Found mapping: {videolink} -> {video_id}")
                        break
            
            if video_id:
                # 检查媒体会话节点是否存在
                media_session_node = f"media_session_{video_id}"
                if media_session_node not in self.G:
                    logger.warning(f"Media session node not found for video_id: {video_id}")
                    continue
                
                # 霸凌标签
                bullying_value = 1 if row['question2'] == 'bullying' else 0
                bullying_label_id = f"label_bullying_{video_id}"
                
                self.G.add_node(
                    bullying_label_id,
                    node_type='Label',
                    target_id=video_id,
                    label_type='bullying',
                    value=bullying_value,
                    confidence=row.get('_golden', 1.0)  # 置信度，默认为1.0
                )
                
                # 添加媒体会话-标签关系
                self.G.add_edge(
                    media_session_node,
                    bullying_label_id,
                    relation='HAS_LABEL',
                    source='human_annotation'
                )
                
                # 攻击性标签
                aggression_value = 1 if row['question1'] == 'aggression' else 0
                aggression_label_id = f"label_aggression_{video_id}"
                
                self.G.add_node(
                    aggression_label_id,
                    node_type='Label',
                    target_id=video_id,
                    label_type='aggression',
                    value=aggression_value,
                    confidence=row.get('_golden', 1.0)  # 置信度，默认为1.0
                )
                
                # 添加媒体会话-标签关系
                self.G.add_edge(
                    media_session_node,
                    aggression_label_id,
                    relation='HAS_LABEL',
                    source='human_annotation'
                )
                
                label_count += 2
                self.stats['label_count'] += 2
                self.stats['relation_count'] += 2
        
        # 构建情感和主题标签
        for _, row in self.data_loader.emotion_data.iterrows():
            videolink = row['videolink']
            video_id = self.get_post_id_from_url(videolink)
            
            # 如果找不到映射，尝试直接匹配视频URL
            if not video_id:
                for v in self.data_loader.video_data:
                    post_id = v.get('_id')
                    url = v.get('url', '')
                    
                    if url and (videolink in url or url.endswith(videolink)):
                        video_id = post_id
                        # 更新映射
                        self.data_loader.videolink_to_postid[videolink] = video_id
                        self.data_loader.postid_to_videolink[video_id] = videolink
                        break
            
            if video_id and f"video_{video_id}" in self.G:
                # 情感标签
                emotion_value = row.get('emotion', 'neutral')
                emotion_label_id = f"label_emotion_{video_id}"
                
                self.G.add_node(
                    emotion_label_id,
                    node_type='Label',
                    target_id=video_id,
                    label_type='emotion',
                    value=emotion_value,
                    confidence=row.get('_trusted', 1.0)  # 置信度，默认为1.0
                )
                
                # 添加视频-标签关系
                self.G.add_edge(
                    f"video_{video_id}",
                    emotion_label_id,
                    relation='HAS_LABEL',
                    source='human_annotation'
                )
                
                # 主题标签
                topic_value = row.get('topic', 'other')
                topic_label_id = f"label_topic_{video_id}"
                
                self.G.add_node(
                    topic_label_id,
                    node_type='Label',
                    target_id=video_id,
                    label_type='topic',
                    value=topic_value,
                    confidence=row.get('_trusted', 1.0)  # 置信度，默认为1.0
                )
                
                # 添加视频-标签关系
                self.G.add_edge(
                    f"video_{video_id}",
                    topic_label_id,
                    relation='HAS_LABEL',
                    source='human_annotation'
                )
                
                label_count += 2
                self.stats['label_count'] += 2
                self.stats['relation_count'] += 2
        
        logger.info(f"Built {self.stats['label_count']} label nodes")
        
        # 如果没有标签节点，输出警告
        if label_count == 0:
            logger.warning("No label nodes were created! Check the mapping between videolink and postId.")
            # 输出一些cyberbullying_data的样本，帮助调试
            if hasattr(self, 'data_loader') and self.data_loader.cyberbullying_data is not None:
                logger.warning("Sample videolinks from cyberbullying_data:")
                for i, row in self.data_loader.cyberbullying_data.head(5).iterrows():
                    logger.warning(f"  {row['videolink']}")
            
            # 输出一些视频URL的样本
            logger.warning("Sample URLs from video_data:")
            for i, video in enumerate(self.data_loader.video_data[:5]):
                logger.warning(f"  {video.get('_id')}: {video.get('url', '')}")

    def build_user_relations(self):
        """构建用户关系"""
        logger.info("Building user relations...")
        relation_count_before = self.stats['relation_count']
        
        # 构建用户-视频发布关系
        for video in self.data_loader.video_data:
            user_id = video.get('username')
            video_id = video['_id']
            
            if user_id and f"user_{user_id}" in self.G and f"video_{video_id}" in self.G:
                self.G.add_edge(
                    f"user_{user_id}",
                    f"video_{video_id}",
                    relation='USER_POSTED_VIDEO',
                    timestamp=video.get('created', '')
                )
                
                self.stats['relation_count'] += 1
        
        # 构建用户-视频评论关系
        for comment in self.data_loader.comment_data:
            user_id = comment.get('username')
            video_id = comment.get('postId')
            
            if user_id and video_id and f"user_{user_id}" in self.G and f"video_{video_id}" in self.G:
                self.G.add_edge(
                    f"user_{user_id}",
                    f"video_{video_id}",
                    relation='USER_COMMENTED_VIDEO',
                    content=comment.get('commentText', ''),
                    timestamp=comment.get('created', ''),
                    comment_id=comment.get('_id', '')
                )
                
                self.stats['relation_count'] += 1
        
        logger.info(f"Built {self.stats['relation_count'] - relation_count_before} user relations")

    def build(self):
        """构建完整的知识图谱"""
        try:
            # 1. 加载数据
            logger.info("Loading data...")
            self.load_data()
            
            # 2. 构建节点
            self.build_video_nodes()
            self.build_user_nodes()
            self.build_media_session_nodes()
            self.build_label_nodes()
            
            # 3. 构建关系
            self.build_user_relations()
            
            # 4. 打印统计信息
            logger.info("\nKnowledge Graph Statistics:")
            logger.info(f"- Video nodes: {self.stats['video_count']}")
            logger.info(f"- User nodes: {self.stats['user_count']}")
            logger.info(f"- Media session nodes: {self.stats['media_session_count']}")
            logger.info(f"- Label nodes: {self.stats['label_count']}")
            logger.info(f"- Total relations: {self.stats['relation_count']}")
            logger.info(f"- Total nodes: {len(self.G.nodes())}")
            logger.info(f"- Average degree: {2*self.stats['relation_count']/len(self.G.nodes()):.2f}")
            
            # 5. 节点类型分布
            node_type_counts = {}
            for node, attrs in self.G.nodes(data=True):
                node_type = attrs.get('node_type', 'Unknown')
                node_type_counts[node_type] = node_type_counts.get(node_type, 0) + 1
            
            logger.info("\nNode Type Distribution:")
            for node_type, count in sorted(node_type_counts.items()):
                logger.info(f"- {node_type}: {count} nodes ({count/len(self.G.nodes()):.2%})")
            
            # 6. 关系类型分布
            relation_type_counts = {}
            for _, _, attrs in self.G.edges(data=True):
                relation_type = attrs.get('relation', 'Unknown')
                relation_type_counts[relation_type] = relation_type_counts.get(relation_type, 0) + 1
            
            logger.info("\nRelation Type Distribution:")
            for relation_type, count in sorted(relation_type_counts.items()):
                logger.info(f"- {relation_type}: {count} relations ({count/self.stats['relation_count']:.2%})")
            
            logger.info("\nKnowledge graph built successfully")
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
            
        except Exception as e:
            logger.error(f"Error saving knowledge graph: {str(e)}")
            raise

    def get_cyberbullying_dataset(self):
        """
        获取用于霸凌检测的数据集
        返回: (features, bullying_labels, aggression_labels)
        """
        # 收集特征和标签
        video_features_list = []
        comment_features_list = []
        bullying_labels = []
        aggression_labels = []
        
        # 记录处理的样本数
        processed_count = 0
        skipped_count = 0
        
        # 遍历所有媒体会话节点
        for node in self.G.nodes():
            if node.startswith('media_session_'):
                node_data = self.G.nodes[node]
                video_id = node_data.get('video_id')
                
                # 获取标签
                bullying_label = None
                aggression_label = None
                
                # 查找关联的标签
                for _, label_node in self.G.out_edges(node):
                    label_data = self.G.nodes[label_node]
                    if label_data.get('node_type') == 'Label':
                        if label_data.get('label_type') == 'bullying':
                            bullying_label = label_data.get('value')
                        elif label_data.get('label_type') == 'aggression':
                            aggression_label = label_data.get('value')
                
                # 只有当两个标签都存在时才添加到数据集
                if bullying_label is not None and aggression_label is not None:
                    # 获取特征
                    session_features = node_data.get('session_features')
                    video_node = f"video_{video_id}"
                    video_features = self.G.nodes[video_node].get('frame_features') if video_node in self.G else None
                    
                    # 检查特征是否存在
                    if video_features is None and session_features is None:
                        logger.warning(f"No features found for media session {node}")
                        skipped_count += 1
                        continue
                    
                    # 确保两种特征都存在
                    if video_features is None or session_features is None:
                        logger.warning(f"Missing one feature type for media session {node}")
                        skipped_count += 1
                        continue
                    
                    # 转换为numpy数组
                    if isinstance(video_features, torch.Tensor):
                        video_features = video_features.cpu().numpy()
                    if isinstance(session_features, torch.Tensor):
                        session_features = session_features.cpu().numpy()
                    
                    # 处理特征维度
                    if len(video_features.shape) > 1:
                        # 如果是多维数组，取平均值
                        video_features = np.mean(video_features, axis=0)
                    if len(session_features.shape) > 1:
                        # 如果是多维数组，取平均值
                        session_features = np.mean(session_features, axis=0)
                    
                    # 添加到特征列表
                    video_features_list.append(video_features)
                    comment_features_list.append(session_features)
                    bullying_labels.append(bullying_label)
                    aggression_labels.append(aggression_label)
                    processed_count += 1
        
        # 检查是否有有效样本
        if processed_count == 0:
            logger.warning("No valid samples found for cyberbullying dataset")
            return np.array([]), np.array([]), np.array([])
        
        # 使用注意力机制融合特征
        logger.info(f"Fusing features for {processed_count} samples using attention mechanism...")
        
        # 获取特征维度
        video_dim = video_features_list[0].shape[0]
        comment_dim = comment_features_list[0].shape[0]
        
        # 初始化融合模型（如果尚未初始化）
        if self.fusion_model is None:
            self.fusion_model = MultimodalFusion(
                video_dim=video_dim, 
                text_dim=comment_dim, 
                output_dim=512
            ).to(self.device)
            logger.info(f"Initialized MultimodalFusion with video_dim={video_dim}, text_dim={comment_dim}")
        
        # 转换为张量
        video_tensor = torch.tensor(np.array(video_features_list), dtype=torch.float32).to(self.device)
        comment_tensor = torch.tensor(np.array(comment_features_list), dtype=torch.float32).to(self.device)
        
        # 使用注意力机制融合特征
        with torch.no_grad():
            fused_features = self.fusion_model(video_tensor, comment_tensor)
            fused_features = fused_features.cpu().numpy()
        
        # 转换标签为numpy数组
        bullying_labels = np.array(bullying_labels)
        aggression_labels = np.array(aggression_labels)
        
        logger.info(f"Processed {processed_count} samples, skipped {skipped_count} samples")
        logger.info(f"Feature vector shape: {fused_features.shape}")
        logger.info(f"Bullying positive samples: {np.sum(bullying_labels)}")
        logger.info(f"Aggression positive samples: {np.sum(aggression_labels)}")
        
        return fused_features, bullying_labels, aggression_labels


if __name__ == "__main__":
    # 加载数据
    data_loader = load_all_data()
    
    # 创建知识图谱构建器
    kg_builder = CyberbullyingKnowledgeGraph(data_loader)
    
    # 构建知识图谱
    G = kg_builder.build()
    
    # 保存知识图谱
    kg_builder.save_graph() 
    
    # 获取霸凌检测数据集
    features, bullying_labels, aggression_labels = kg_builder.get_cyberbullying_dataset()
    logger.info(f"\nCyberbullying Dataset:")
    logger.info(f"- Features shape: {features.shape}")
    logger.info(f"- Bullying labels shape: {bullying_labels.shape}")
    logger.info(f"- Aggression labels shape: {aggression_labels.shape}")
    logger.info(f"- Bullying positive samples: {np.sum(bullying_labels)}")
    logger.info(f"- Aggression positive samples: {np.sum(aggression_labels)}")
