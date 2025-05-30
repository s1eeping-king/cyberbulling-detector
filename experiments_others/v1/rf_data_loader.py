import numpy as np
import pandas as pd
from neo4j import GraphDatabase
from typing import Dict, List, Tuple, Optional
import json
import os
import torch

class RFDataLoader:
    """从Neo4j加载数据并处理为随机森林分类器所需格式的数据加载器"""

    def __init__(self, uri: str, user: str, password: str, debug: bool = False):
        """初始化数据加载器"""
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.debug = debug  # 调试模式标志
        self.debug_counter = 0  # 调试计数器

        # 加载BERT特征
        self.comment_bert_features = self._load_bert_features("data/processed/bert_features/comment_bert_features.json", "comment")
        
        # 加载视频描述情感分析结果
        self.video_sentiment = self._load_video_sentiment("data/processed/roberta_classification/video_descriptions_sentiment.json")
        
        # 加载评论情感分析结果
        self.comment_sentiment = self._load_comment_sentiment("data/processed/roberta_classification/comments_sentiment.json")

        print(f"已加载 {len(self.comment_bert_features)} 条评论BERT特征")
        print(f"已加载 {len(self.video_sentiment)} 个视频描述情感分析结果")
        print(f"已加载 {len(self.comment_sentiment)} 条评论情感分析结果")

    def _load_bert_features(self, file_path: str, feature_type: str) -> Dict[str, np.ndarray]:
        """加载BERT特征

        Args:
            file_path: BERT特征文件路径
            feature_type: 特征类型 ('comment' 或 'media')

        Returns:
            Dict[str, np.ndarray]: ID到特征的映射
        """
        if not os.path.exists(file_path):
            print(f"警告: BERT特征文件不存在: {file_path}")
            return {}

        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        features_dict = {}

        if feature_type == "comment":
            ids = data.get("comment_ids", [])
            features = data.get("bert_features", [])
        else:  # media
            ids = data.get("media_ids", [])
            features = data.get("bert_features", [])

        for id_, feature in zip(ids, features):
            features_dict[id_] = np.array(feature, dtype=np.float32)

        return features_dict
    
    def _load_video_sentiment(self, file_path: str) -> Dict[str, Dict]:
        """加载视频描述情感分析结果

        Args:
            file_path: 情感分析结果文件路径

        Returns:
            Dict[str, Dict]: ID到情感分析结果的映射
        """
        if not os.path.exists(file_path):
            print(f"警告: 视频描述情感分析文件不存在: {file_path}")
            return {}

        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        sentiment_dict = {}
        for item in data:
            post_id = item.get("postId")
            if post_id:
                sentiment_dict[post_id] = {
                    "sentiment": item.get("sentiment", "neutral"),
                    "confidence": item.get("confidence", 0.0)
                }

        return sentiment_dict
    
    def _load_comment_sentiment(self, file_path: str) -> Dict[str, Dict]:
        """加载评论情感分析结果

        Args:
            file_path: 情感分析结果文件路径

        Returns:
            Dict[str, Dict]: ID到情感分析结果的映射
        """
        if not os.path.exists(file_path):
            print(f"警告: 评论情感分析文件不存在: {file_path}")
            return {}

        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        sentiment_dict = {}
        for item in data:
            comment_id = item.get("commentId")
            if comment_id:
                sentiment_dict[comment_id] = {
                    "sentiment": item.get("sentiment", "neutral"),
                    "confidence": item.get("confidence", 0.0)
                }

        return sentiment_dict

    def get_all_media_sessions(self) -> List[str]:
        """获取所有媒体会话ID"""
        query = """
        MATCH (m:MediaSession)
        RETURN m.id AS id
        """
        with self.driver.session() as session:
            result = session.run(query)
            return [record["id"] for record in result]

    def load_media_session_features(self, media_session_id: str) -> Optional[Dict]:
        """加载单个媒体会话的特征

        Args:
            media_session_id: 媒体会话ID

        Returns:
            Dict: 包含媒体会话特征的字典，如果未找到则返回None
        """
        query = """
        MATCH (center:MediaSession {id: $media_id})
        OPTIONAL MATCH (center)-[:HAS_LABEL]->(l:Label)
        RETURN center, collect(distinct l) as labels
        """

        with self.driver.session() as session:
            result = session.run(query, media_id=media_session_id).single()
            if not result:
                return None

            # 获取中心媒体会话节点
            center = result['center']
            labels = result['labels']
            
            # 处理标签
            bullying = 0
            if labels:
                properties = dict(labels[0])
                bullying = 1 if properties.get('value', 'noneBll') == 'bullying' else 0
            
            # 获取媒体会话属性
            properties = dict(center)
            
            # 获取视频描述情感
            sentiment_info = self.video_sentiment.get(media_session_id, {"sentiment": "neutral", "confidence": 0.0})
            sentiment = sentiment_info["sentiment"]
            sentiment_confidence = sentiment_info["confidence"]
            
            # 将情感转换为数值
            sentiment_map = {
                "positive": 1.0,
                "neutral": 0.0,
                "negative": -1.0
            }
            sentiment_value = sentiment_map.get(sentiment, 0.0)
            
            # 构建特征字典
            features = {
                "media_id": media_session_id,
                "like_count": float(properties.get('likeCount', 0)),
                "comment_count": float(properties.get('commentCount', 0)),
                "loop_count": float(properties.get('loopCount', 0)),
                "repost_count": float(properties.get('repostCount', 0)),
                "description_sentiment": sentiment_value,
                "description_sentiment_confidence": sentiment_confidence,
                "label": bullying
            }
            
            return features

    def load_user_features(self, media_session_id: str) -> Dict[str, Dict]:
        """加载与媒体会话相关的用户特征

        Args:
            media_session_id: 媒体会话ID

        Returns:
            Dict[str, Dict]: 用户ID到特征的映射
        """
        query = """
        MATCH (m:MediaSession {id: $media_id})-[r]-(u:User)
        RETURN u
        """

        with self.driver.session() as session:
            result = session.run(query, media_id=media_session_id)
            users = {}
            
            for record in result:
                user = record["u"]
                properties = dict(user)
                user_id = properties.get('id', '')
                
                if user_id:
                    users[user_id] = {
                        "follower_count": float(properties.get('followerCount', 0)),
                        "following_count": float(properties.get('followingCount', 0)),
                        "post_count": float(properties.get('postCount', 0))
                    }
            
            return users

    def load_comment_features(self, media_session_id: str) -> Dict[str, Dict]:
        """加载与媒体会话相关的评论特征

        Args:
            media_session_id: 媒体会话ID

        Returns:
            Dict[str, Dict]: 评论ID到特征的映射
        """
        query = """
        MATCH (m:MediaSession {id: $media_id})-[r]-(c:Comment)
        RETURN c
        """

        with self.driver.session() as session:
            result = session.run(query, media_id=media_session_id)
            comments = {}
            
            for record in result:
                comment = record["c"]
                properties = dict(comment)
                comment_id = properties.get('id', '')
                
                if comment_id:
                    # 获取评论情感
                    sentiment_info = self.comment_sentiment.get(comment_id, {"sentiment": "neutral", "confidence": 0.0})
                    sentiment = sentiment_info["sentiment"]
                    sentiment_confidence = sentiment_info["confidence"]
                    
                    # 将情感转换为数值
                    sentiment_map = {
                        "positive": 1.0,
                        "neutral": 0.0,
                        "negative": -1.0
                    }
                    sentiment_value = sentiment_map.get(sentiment, 0.0)
                    
                    # 获取BERT特征
                    bert_features = self.comment_bert_features.get(comment_id, np.zeros(768, dtype=np.float32))
                    
                    comments[comment_id] = {
                        "sentiment": sentiment_value,
                        "sentiment_confidence": sentiment_confidence,
                        "bert_features": bert_features
                    }
            
            return comments

    def aggregate_features(self, media_session_id: str) -> Optional[Dict]:
        """聚合媒体会话、用户和评论特征

        Args:
            media_session_id: 媒体会话ID

        Returns:
            Dict: 聚合后的特征字典，如果未找到则返回None
        """
        # 加载媒体会话特征
        media_features = self.load_media_session_features(media_session_id)
        if not media_features:
            return None
        
        # 加载用户特征
        user_features = self.load_user_features(media_session_id)
        
        # 加载评论特征
        comment_features = self.load_comment_features(media_session_id)
        
        # 聚合用户特征
        if user_features:
            avg_follower_count = np.mean([u["follower_count"] for u in user_features.values()])
            avg_following_count = np.mean([u["following_count"] for u in user_features.values()])
            avg_post_count = np.mean([u["post_count"] for u in user_features.values()])
        else:
            avg_follower_count = 0.0
            avg_following_count = 0.0
            avg_post_count = 0.0
        
        # 聚合评论特征
        if comment_features:
            avg_comment_sentiment = np.mean([c["sentiment"] for c in comment_features.values()])
            # 计算正面、中性和负面评论的比例
            positive_ratio = sum(1 for c in comment_features.values() if c["sentiment"] > 0) / len(comment_features)
            neutral_ratio = sum(1 for c in comment_features.values() if c["sentiment"] == 0) / len(comment_features)
            negative_ratio = sum(1 for c in comment_features.values() if c["sentiment"] < 0) / len(comment_features)
            
            # 聚合BERT特征 (使用平均值)
            avg_bert_features = np.mean([c["bert_features"] for c in comment_features.values()], axis=0)
        else:
            avg_comment_sentiment = 0.0
            positive_ratio = 0.0
            neutral_ratio = 0.0
            negative_ratio = 0.0
            avg_bert_features = np.zeros(768, dtype=np.float32)
        
        # 合并所有特征
        aggregated_features = {
            "media_id": media_session_id,
            # 媒体会话特征
            "like_count": media_features["like_count"],
            "comment_count": media_features["comment_count"],
            "loop_count": media_features["loop_count"],
            "repost_count": media_features["repost_count"],
            "description_sentiment": media_features["description_sentiment"],
            "description_sentiment_confidence": media_features["description_sentiment_confidence"],
            # 用户特征
            "avg_follower_count": avg_follower_count,
            "avg_following_count": avg_following_count,
            "avg_post_count": avg_post_count,
            # 评论特征
            "avg_comment_sentiment": avg_comment_sentiment,
            "positive_comment_ratio": positive_ratio,
            "neutral_comment_ratio": neutral_ratio,
            "negative_comment_ratio": negative_ratio,
            # 标签
            "label": media_features["label"]
        }
        
        # 添加BERT特征
        for i, value in enumerate(avg_bert_features):
            aggregated_features[f"bert_{i}"] = value
        
        return aggregated_features

    def load_dataset(self, media_session_ids: Optional[List[str]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """加载数据集

        Args:
            media_session_ids: 媒体会话ID列表，如果为None则加载所有媒体会话

        Returns:
            Tuple[np.ndarray, np.ndarray]: 特征矩阵和标签向量
        """
        if media_session_ids is None:
            media_session_ids = self.get_all_media_sessions()
        
        features_list = []
        
        for media_id in media_session_ids:
            features = self.aggregate_features(media_id)
            if features:
                features_list.append(features)
        
        if not features_list:
            return np.array([]), np.array([])
        
        # 转换为DataFrame
        df = pd.DataFrame(features_list)
        
        # 分离特征和标签
        X = df.drop(columns=["media_id", "label"])
        y = df["label"].values
        
        return X.values, y

    def close(self):
        """关闭数据库连接"""
        self.driver.close()
