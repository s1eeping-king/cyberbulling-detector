import torch
from torch_geometric.data import HeteroData
from neo4j import GraphDatabase
from typing import Dict, List, Tuple, Optional
import json
import os
import numpy as np
import pandas as pd

class DataLoader:
    """从Neo4j加载数据并处理为PyG格式的数据加载器"""

    def __init__(self, uri: str, user: str, password: str, device: torch.device, debug: bool = False):
        """初始化数据加载器"""
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.device = device
        self.debug = debug  # 调试模式标志
        self.debug_counter = 0  # 调试计数器

        # 加载RoBERTa特征
        self.comment_roberta_features = self._load_roberta_features("data/processed/RoBERTa_features/comment_roberta_features.json", "comment")
        self.media_roberta_features = self._load_roberta_features("data/processed/RoBERTa_features/media_roberta_features.json", "media")

        # 加载VADER情感分数
        self.comment_sentiment_scores = self._load_sentiment_scores("data/processed/vader_sentiment_scores/comment_sentiment_scores.json", "comment")
        self.media_sentiment_scores = self._load_sentiment_scores("data/processed/vader_sentiment_scores/media_sentiment_scores.json", "media")

        # 加载用户特征
        self.train_user_features = self._load_user_features("data/processed/features_extended/train_features.csv")
        self.val_user_features = self._load_user_features("data/processed/features_extended/val_features.csv")
        self.test_user_features = self._load_user_features("data/processed/features_extended/test_features.csv")

        # 加载数据集划分
        self.train_sessions = self._load_json_list("data/processed/splits/train_sessions.json")
        self.val_sessions = self._load_json_list("data/processed/splits/val_sessions.json")
        self.test_sessions = self._load_json_list("data/processed/splits/test_sessions.json")
        self.train_users = self._load_json_list("data/processed/splits/train_users.json")
        self.val_users = self._load_json_list("data/processed/splits/val_users.json")
        self.test_users = self._load_json_list("data/processed/splits/test_users.json")

        print(f"已加载 {len(self.comment_roberta_features)} 条评论RoBERTa特征")
        print(f"已加载 {len(self.media_roberta_features)} 个媒体会话RoBERTa特征")
        print(f"已加载 {len(self.comment_sentiment_scores)} 条评论VADER情感分数")
        print(f"已加载 {len(self.media_sentiment_scores)} 个媒体会话VADER情感分数")
        print(f"已加载 {len(self.train_user_features)} 个训练集用户特征")
        print(f"已加载 {len(self.val_user_features)} 个验证集用户特征")
        print(f"已加载 {len(self.test_user_features)} 个测试集用户特征")
        print(f"已加载数据集划分: 训练集 {len(self.train_sessions)} 个会话, 验证集 {len(self.val_sessions)} 个会话, 测试集 {len(self.test_sessions)} 个会话")

    def _load_roberta_features(self, file_path: str, feature_type: str) -> Dict[str, torch.Tensor]:
        """加载RoBERTa特征

        Args:
            file_path: RoBERTa特征文件路径
            feature_type: 特征类型 ('comment' 或 'media')

        Returns:
            Dict[str, torch.Tensor]: ID到特征的映射
        """
        if not os.path.exists(file_path):
            print(f"警告: RoBERTa特征文件不存在: {file_path}")
            return {}

        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        features_dict = {}

        if feature_type == "comment":
            ids = data.get("comment_ids", [])
            features = data.get("roberta_features", [])
        else:  # media
            ids = data.get("media_ids", [])
            features = data.get("roberta_features", [])

        for id_, feature in zip(ids, features):
            features_dict[id_] = torch.tensor(feature, dtype=torch.float)

        return features_dict

    def _load_sentiment_scores(self, file_path: str, feature_type: str) -> Dict[str, Dict[str, float]]:
        """加载VADER情感分数

        Args:
            file_path: VADER情感分数文件路径
            feature_type: 特征类型 ('comment' 或 'media')

        Returns:
            Dict[str, Dict[str, float]]: ID到情感分数的映射
        """
        if not os.path.exists(file_path):
            print(f"警告: VADER情感分数文件不存在: {file_path}")
            return {}

        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        sentiment_dict = {}

        if feature_type == "comment":
            ids = data.get("comment_ids", [])
            scores = data.get("sentiment_scores", [])
        else:  # media
            ids = data.get("media_ids", [])
            scores = data.get("sentiment_scores", [])

        for id_, score in zip(ids, scores):
            # 只保留pos、neg和compound三个维度
            sentiment_dict[id_] = {
                'pos': float(score.get('pos', 0.0)),
                'neg': float(score.get('neg', 0.0)),
                'compound': float(score.get('compound', 0.0))
            }

        return sentiment_dict

    def _load_user_features(self, file_path: str) -> Dict[str, Dict[str, float]]:
        """加载用户特征

        Args:
            file_path: 用户特征CSV文件路径

        Returns:
            Dict[str, Dict[str, float]]: 用户ID到特征字典的映射
        """
        if not os.path.exists(file_path):
            print(f"警告: 用户特征文件不存在: {file_path}")
            return {}

        try:
            # 读取CSV文件
            df = pd.read_csv(file_path)

            # 将DataFrame转换为字典
            features_dict = {}
            for _, row in df.iterrows():
                user_id = str(row['userId'])
                features = {col: float(row[col]) for col in df.columns if col != 'userId' and pd.notna(row[col])}
                features_dict[user_id] = features

            return features_dict
        except Exception as e:
            print(f"加载用户特征时出错: {str(e)}")
            return {}

    def _load_json_list(self, file_path: str) -> List[str]:
        """加载JSON列表文件

        Args:
            file_path: JSON文件路径

        Returns:
            List[str]: 字符串列表
        """
        if not os.path.exists(file_path):
            print(f"警告: JSON文件不存在: {file_path}")
            return []

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return [str(item) for item in data]
        except Exception as e:
            print(f"加载JSON文件时出错: {str(e)}")
            return []

    def get_all_media_sessions(self) -> List[str]:
        """获取所有媒体会话ID"""
        query = """
        MATCH (m:MediaSession)
        RETURN m.id AS id
        """
        with self.driver.session() as session:
            result = session.run(query)
            return [record["id"] for record in result]

    def load_subgraph(self, media_session_id: str) -> HeteroData:
        """加载单个媒体会话的子图，使用混合图结构（同时包含评论节点和用户间直接边）"""
        query = """
        MATCH (center:MediaSession {id: $media_id})-[r1]-(neighbor)
        WITH center, collect(distinct neighbor) as neighbors
        UNWIND neighbors as n
        MATCH (n)-[r]-(other)
        WHERE other IN neighbors OR other = center
        RETURN center,
               // 节点
               collect(distinct n) as all_neighbors,
               collect(distinct {start: startNode(r), end: endNode(r), type: type(r)}) as internal_relationships
        """

        with self.driver.session() as session:
            result = session.run(query, media_id=media_session_id).single()
            if not result:
                return None

            data = HeteroData()

            # 获取中心媒体会话节点
            center = result['center']

            # 获取所有邻居节点
            all_neighbors = result['all_neighbors']

            # 获取所有内部关系
            internal_relationships = result['internal_relationships']

            # 分类节点
            users = []
            comments = []
            media_sessions = [center]  # 中心媒体会话节点

            # 处理所有邻居节点
            for node in all_neighbors:
                labels = list(node.labels)
                if 'User' in labels:
                    users.append(node)
                elif 'Comment' in labels:
                    comments.append(node)
                elif 'MediaSession' in labels:
                    media_sessions.append(node)

            # 1. 处理用户节点
            user_features, user_id_to_idx = self._process_user_features(users)
            data['user'].x = user_features

            # 2. 处理媒体会话节点
            # 只处理中心媒体会话节点，忽略其他媒体会话节点
            data['media_session'].x = self._process_media_features(center)

            # 3. 处理评论节点
            comment_features, comment_id_to_idx = self._process_comment_features(comments)
            data['comment'].x = comment_features

            # 4. 处理关系
            # 初始化边列表
            publishes_edges = []
            creates_edges = []
            belongs_to_edges = []
            mentions_edges = []
            offensive_comment_edges = []
            non_offensive_comment_edges = []

            # 处理所有内部关系
            for rel in internal_relationships:
                start_node = rel['start']
                end_node = rel['end']
                rel_type = rel['type']

                start_id = str(start_node.element_id)
                end_id = str(end_node.element_id)

                # 根据关系类型分类
                if rel_type == 'PUBLISHES':
                    if start_id in user_id_to_idx:  # 确保起始节点是用户
                        publishes_edges.append([start_id, end_id])

                elif rel_type == 'CREATES':
                    if start_id in user_id_to_idx and end_id in comment_id_to_idx:  # 用户创建评论
                        creates_edges.append([start_id, end_id])

                elif rel_type == 'BELONGS_TO':
                    if start_id in comment_id_to_idx:  # 评论属于媒体会话
                        belongs_to_edges.append([start_id, end_id])

                elif rel_type == 'MENTIONS':
                    if start_id in comment_id_to_idx and end_id in user_id_to_idx:  # 评论提到用户
                        mentions_edges.append([start_id, end_id])

                elif rel_type == 'OFFENSIVE_COMMENT':
                    if start_id in user_id_to_idx and end_id in user_id_to_idx:  # 用户间攻击性评论
                        offensive_comment_edges.append([start_id, end_id, ''])  # 添加空的comment_id

                elif rel_type == 'NON_OFFENSIVE_COMMENT':
                    if start_id in user_id_to_idx and end_id in user_id_to_idx:  # 用户间非攻击性评论
                        non_offensive_comment_edges.append([start_id, end_id, ''])  # 添加空的comment_id

            # 处理原始关系的边索引
            # 用户-发布->媒体会话
            data['user', 'publishes', 'media_session'].edge_index = self._process_edge_index(
                publishes_edges,
                {'user': data['user'].x.shape[0], 'media_session': data['media_session'].x.shape[0]},
                user_id_to_idx
            )

            # 用户-创建->评论
            data['user', 'creates', 'comment'].edge_index = self._process_edge_index(
                creates_edges,
                {'user': data['user'].x.shape[0], 'comment': data['comment'].x.shape[0]},
                user_id_to_idx,
                dst_id_to_idx=comment_id_to_idx
            )

            # 评论-属于->媒体会话
            data['comment', 'belongs_to', 'media_session'].edge_index = self._process_edge_index(
                belongs_to_edges,
                {'comment': data['comment'].x.shape[0], 'media_session': data['media_session'].x.shape[0]},
                comment_id_to_idx
            )

            # 评论-提到->用户
            data['comment', 'mentions', 'user'].edge_index = self._process_edge_index(
                mentions_edges,
                {'comment': data['comment'].x.shape[0], 'user': data['user'].x.shape[0]},
                comment_id_to_idx,
                dst_id_to_idx=user_id_to_idx
            )

            # 处理用户间直接边
            # 攻击性评论边
            offensive_edge_index = self._process_direct_comment_edge(
                offensive_comment_edges,
                {'user': data['user'].x.shape[0]},
                user_id_to_idx
            )
            data['user', 'offensive_comment', 'user'].edge_index = offensive_edge_index

            # 非攻击性评论边
            non_offensive_edge_index = self._process_direct_comment_edge(
                non_offensive_comment_edges,
                {'user': data['user'].x.shape[0]},
                user_id_to_idx
            )
            data['user', 'non_offensive_comment', 'user'].edge_index = non_offensive_edge_index

            return data



    def _process_media_features(self, media) -> torch.Tensor:
        """处理媒体会话节点特征，整合RoBERTa特征和VADER情感分数"""
        if not media:
            # 返回一个占位符特征，维度为(1, 10+768+3)，包括原始特征、RoBERTa特征和VADER情感分数
            return torch.zeros((1, 781), dtype=torch.float, device=self.device)

        properties = dict(media)
        media_id = properties.get('id', '')

        # 处理字符串类型的属性
        description_offensive = properties.get('description_offensive', 'not_offensive')
        description_offensive_value = 1.0 if description_offensive == 'offensive' else 0.0

        # 处理emotion和theme
        emotion_map = {
            'neutral': 0.0,
            'joy': 1.0,
            'sad': 2.0,
            'love': 3.0,
            'surprise': 4.0,
            'fear': 5.0,
            'anger': 6.0
        }

        theme_map = {
            'other': 0.0,
            'people': 1.0,
            'person': 2.0,
            'indoor': 3.0,
            'outdoor': 4.0,
            'cartoon': 5.0,
            'text': 6.0,
            'activity': 7.0,
            'animal': 8.0
        }

        emotion = properties.get('emotion', 'neutral')
        emotion_value = emotion_map.get(emotion, 0.0)

        theme = properties.get('theme', 'other')
        theme_value = theme_map.get(theme, 0.0)

        # 获取原始特征
        original_features = torch.tensor([
            float(properties.get('commentCount_normalized', 0.0)),
            description_offensive_value,
            float(properties.get('description_offensive_confidence', 0.0)),
            emotion_value,
            float(properties.get('emotion_confidence', 0.0)),
            float(properties.get('likeCount_normalized', 0.0)),
            float(properties.get('loopCount_normalized', 0.0)),
            float(properties.get('repostCount_normalized', 0.0)),
            theme_value,
            float(properties.get('theme_confidence', 0.0))
        ], device=self.device)

        # 获取RoBERTa特征
        if media_id in self.media_roberta_features:
            roberta_features = self.media_roberta_features[media_id].to(self.device)
        else:
            # 如果没有找到RoBERTa特征，使用零向量
            roberta_features = torch.zeros(768, device=self.device)

        # 获取VADER情感分数
        if media_id in self.media_sentiment_scores:
            sentiment_scores = self.media_sentiment_scores[media_id]
            sentiment_features = torch.tensor([
                sentiment_scores['pos'],
                sentiment_scores['neg'],
                sentiment_scores['compound']
            ], device=self.device)
        else:
            # 如果没有找到VADER情感分数，使用零向量
            sentiment_features = torch.zeros(3, device=self.device)

        # 合并原始特征、RoBERTa特征和VADER情感分数
        features = torch.cat([original_features, roberta_features, sentiment_features])

        # 确保是二维张量
        if features.dim() == 1:
            features = features.unsqueeze(0)
        return features

    def _process_comment_features(self, comments) -> Tuple[torch.Tensor, Dict[str, int]]:
        """处理评论节点特征，返回特征矩阵和ID到索引的映射，整合RoBERTa特征和VADER情感分数"""
        if not comments:
            return torch.zeros((0, 773), dtype=torch.float, device=self.device), {}

        # 使用字典进行去重，以评论elementId为键
        unique_comments = {}
        id_to_idx = {}
        current_idx = 0

        for comment in comments:
            properties = dict(comment)
            comment_id = str(comment.element_id)  # 使用节点的element_id属性

            if comment_id not in unique_comments:
                # 处理字符串类型的属性
                offensive = properties.get('offensive', 'not_offensive')
                offensive_value = 1.0 if offensive == 'offensive' else 0.0
                confidence = float(properties.get('offensive_confidence', 1.0))

                # 基本特征
                basic_features = torch.tensor([
                    offensive_value,
                    confidence
                ], device=self.device)

                # RoBERTa特征
                comment_db_id = properties.get('id', '')
                if comment_db_id in self.comment_roberta_features:
                    roberta_features = self.comment_roberta_features[comment_db_id].to(self.device)
                else:
                    roberta_features = torch.zeros(768, device=self.device)

                # VADER情感分数
                if comment_db_id in self.comment_sentiment_scores:
                    sentiment_scores = self.comment_sentiment_scores[comment_db_id]
                    sentiment_features = torch.tensor([
                        sentiment_scores['pos'],
                        sentiment_scores['neg'],
                        sentiment_scores['compound']
                    ], device=self.device)
                else:
                    # 如果没有找到VADER情感分数，使用零向量
                    sentiment_features = torch.zeros(3, device=self.device)

                # 合并特征：基本特征 + RoBERTa特征 + VADER情感分数
                features = torch.cat([basic_features, roberta_features, sentiment_features])

                unique_comments[comment_id] = features
                id_to_idx[comment_id] = current_idx
                current_idx += 1

        # 将去重后的特征转换为tensor
        features_list = list(unique_comments.values())
        if not features_list:
            return torch.zeros((0, 773), dtype=torch.float, device=self.device), {}

        return torch.stack(features_list), id_to_idx

    def _process_direct_comment_edge(self, edge_data, node_counts, user_id_to_idx) -> torch.Tensor:
        """处理用户间直接评论边索引"""
        if not edge_data:
            return torch.zeros((2, 0), dtype=torch.long, device=self.device)

        edge_indices = []

        for e in edge_data:
            if len(e) >= 3 and e[0] is not None and e[1] is not None and e[2] is not None:
                src_id = str(e[0])  # 评论者
                dst_id = str(e[1])  # 被评论者
                # comment_id不再使用

                # 处理源节点和目标节点索引
                if src_id in user_id_to_idx and dst_id in user_id_to_idx:
                    src_idx = user_id_to_idx[src_id]
                    dst_idx = user_id_to_idx[dst_id]

                    # 验证索引是否有效
                    if (src_idx < node_counts['user'] and dst_idx < node_counts['user']):
                        edge_indices.append([src_idx, dst_idx])

        if not edge_indices:
            return torch.zeros((2, 0), dtype=torch.long, device=self.device)

        return torch.tensor(edge_indices, dtype=torch.long, device=self.device).t()

    def _process_user_features(self, users) -> Tuple[torch.Tensor, Dict[str, int]]:
        """处理用户节点特征，返回特征矩阵和ID到索引的映射"""
        if not users:
            return torch.zeros((0, 34), dtype=torch.float, device=self.device), {}  # 增加特征维度

        # 使用字典进行去重，以用户elementId为键
        unique_users = {}
        id_to_idx = {}
        current_idx = 0

        for user in users:
            properties = dict(user)
            # 获取用户ID
            user_element_id = str(user.element_id)  # 使用节点的element_id属性
            user_db_id = properties.get('id', '')  # 数据库中的ID

            if user_element_id not in unique_users:
                # 基本特征
                description_offensive = properties.get('description_offensive', 'not_offensive')
                description_offensive_value = 1.0 if description_offensive == 'offensive' else 0.0

                basic_features = [
                    description_offensive_value,
                    float(properties.get('followerCount_normalized', 0.0)),
                    float(properties.get('followingCount_normalized', 0.0)),
                    float(properties.get('likeCount_normalized', 0.0)),
                    float(properties.get('postCount_normalized', 0.0))
                ]

                # 尝试从扩展特征中获取更多特征
                extended_features = []

                # 根据媒体会话ID确定使用哪个特征集
                media_id = self._get_media_session_for_user(user_element_id)

                if media_id in self.train_sessions:
                    user_features_dict = self.train_user_features
                elif media_id in self.val_sessions:
                    user_features_dict = self.val_user_features
                else:
                    user_features_dict = self.test_user_features

                # 尝试获取扩展特征
                if user_db_id in user_features_dict:
                    # 获取扩展特征
                    feature_dict = user_features_dict[user_db_id]

                    # 添加重要的特征
                    extended_features = [
                        float(feature_dict.get('mention_count', 0.0)),
                        float(feature_dict.get('received_mention_count', 0.0)),
                        float(feature_dict.get('received_offensive_count', 0.0)),
                        float(feature_dict.get('received_non_offensive_count', 0.0)),
                        float(feature_dict.get('offensive_mention_count', 0.0)),
                        float(feature_dict.get('non_offensive_mention_count', 0.0)),
                        float(feature_dict.get('unique_mentioners_count', 0.0)),
                        float(feature_dict.get('unique_targets_count', 0.0)),
                        float(feature_dict.get('max_mentions_to_single_user', 0.0)),
                        float(feature_dict.get('max_offensive_to_single_user', 0.0)),
                        float(feature_dict.get('max_non_offensive_to_single_user', 0.0)),
                        float(feature_dict.get('max_offensive_from_single_user', 0.0)),
                        float(feature_dict.get('max_non_offensive_from_single_user', 0.0)),
                        float(feature_dict.get('bidirectional_offensive_users', 0.0)),
                        float(feature_dict.get('bidirectional_non_offensive_users', 0.0)),
                        float(feature_dict.get('unretaliated_ratio', 0.0)),
                        float(feature_dict.get('comment_count', 0.0)),
                        float(feature_dict.get('offensive_comment_count', 0.0)),
                        float(feature_dict.get('non_offensive_comment_count', 0.0)),
                        float(feature_dict.get('offensive_comment_ratio', 0.0)),
                        float(feature_dict.get('media_session_count', 0.0)),
                        float(feature_dict.get('avg_comments_per_session', 0.0)),
                        float(feature_dict.get('avg_offensive_comments_per_session', 0.0)),
                        float(feature_dict.get('offensive_sessions_ratio', 0.0)),
                        float(feature_dict.get('offensive_comment_frequency', 0.0)),
                        float(feature_dict.get('first_comment_offensive_ratio', 0.0)),
                        float(feature_dict.get('last_comment_offensive_ratio', 0.0)),
                        float(feature_dict.get('offensive_mention_ratio', 0.0)),
                        float(feature_dict.get('non_offensive_mention_ratio', 0.0))
                    ]

                # 如果没有扩展特征，使用零向量
                if not extended_features:
                    extended_features = [0.0] * 29  # 29个扩展特征

                # 合并基本特征和扩展特征
                all_features = basic_features + extended_features

                # 转换为张量
                features = torch.tensor(all_features, device=self.device)

                unique_users[user_element_id] = features
                id_to_idx[user_element_id] = current_idx
                current_idx += 1

        # 将去重后的特征转换为tensor
        features_list = list(unique_users.values())
        if not features_list:
            return torch.zeros((0, 34), dtype=torch.float, device=self.device), {}

        return torch.stack(features_list), id_to_idx

    def _get_media_session_for_user(self, user_element_id: str) -> str:
        """获取用户所在的媒体会话ID"""
        query = """
        MATCH (u:User)-[:CREATES]->(:Comment)-[:BELONGS_TO]->(m:MediaSession)
        WHERE id(u) = $user_element_id
        RETURN m.id AS media_id
        LIMIT 1
        """

        try:
            with self.driver.session() as session:
                result = session.run(query, user_element_id=int(user_element_id)).single()
                if result:
                    return result["media_id"]
        except Exception as e:
            if self.debug:
                print(f"获取用户所在媒体会话时出错: {str(e)}")

        return ""

    def _process_edge_index(self, edge_data, node_counts, src_id_to_idx=None, dst_id_to_idx=None) -> torch.Tensor:
        """处理边索引，使用节点计数和ID映射"""
        if not edge_data:
            return torch.zeros((2, 0), dtype=torch.long, device=self.device)

        edge_indices = []
        for e in edge_data:
            if e[0] is not None and e[1] is not None:
                src_id = str(e[0])
                dst_id = str(e[1])

                # 处理源节点索引
                if src_id_to_idx is not None and src_id in src_id_to_idx:
                    src_idx = src_id_to_idx[src_id]
                else:
                    # 对于没有映射的节点，使用简单的哈希映射
                    src_idx = hash(src_id) % node_counts[list(node_counts.keys())[0]]

                # 处理目标节点索引
                if dst_id_to_idx is not None and dst_id in dst_id_to_idx:
                    dst_idx = dst_id_to_idx[dst_id]
                elif src_id_to_idx is not None and dst_id in src_id_to_idx:
                    # 如果没有提供dst_id_to_idx，但dst_id在src_id_to_idx中
                    dst_idx = src_id_to_idx[dst_id]
                else:
                    # 对于没有映射的节点，使用简单的哈希映射
                    dst_idx = hash(dst_id) % node_counts[list(node_counts.keys())[1]]

                # 验证索引是否有效
                if (src_idx < node_counts[list(node_counts.keys())[0]] and
                    dst_idx < node_counts[list(node_counts.keys())[1]]):
                    edge_indices.append([src_idx, dst_idx])

        if not edge_indices:
            return torch.zeros((2, 0), dtype=torch.long, device=self.device)

        return torch.tensor(edge_indices, dtype=torch.long, device=self.device).t()

    def get_label(self, media_session_id: str) -> int:
        """获取媒体会话的标签（0或1）

        Args:
            media_session_id: 媒体会话ID

        Returns:
            int: 标签值，1表示霸凌，0表示非霸凌
        """
        query = """
        MATCH (m:MediaSession {id: $media_id})-[:HAS_LABEL]->(l:Label)
        RETURN collect(distinct l) as labels
        """

        with self.driver.session() as session:
            result = session.run(query, media_id=media_session_id).single()
            if not result:
                return 0  # 默认为非霸凌

            labels = result['labels']
            if not labels:
                return 0  # 默认为非霸凌

            properties = dict(labels[0])
            bullying = 1 if properties.get('value', 'noneBll') == 'bullying' else 0

            return bullying

    def get_labels(self, media_session_id: str) -> torch.Tensor:
        """获取媒体会话的标签"""
        query = """
        MATCH (m:MediaSession {id: $media_id})-[:HAS_LABEL]->(l:Label)
        RETURN collect(distinct l) as labels
        """

        with self.driver.session() as session:
            result = session.run(query, media_id=media_session_id).single()
            if not result:
                if not hasattr(self, '_warning_count'):
                    self._warning_count = 0
                self._warning_count += 1
                if self._warning_count <= 3:  # 只显示前3个警告
                    print(f"警告: 媒体会话 {media_session_id} 未找到标签数据")
                return torch.zeros((1,), dtype=torch.float, device=self.device)

            labels = result['labels']
            if not labels:
                return torch.zeros((1,), dtype=torch.float, device=self.device)

            properties = dict(labels[0])
            bullying = float(properties.get('value', 'noneBll') == 'bullying')

            return torch.tensor([bullying], dtype=torch.float, device=self.device)

    def close(self):
        """关闭数据库连接"""
        self.driver.close()

    def load_batch(self, media_session_ids: List[str]) -> Dict:
        """批量加载数据
        Args:
            media_session_ids: 媒体会话ID列表
        Returns:
            Dict: 包含合并后的图数据和标签的字典
        """
        batch_graphs = []
        batch_labels = []

        # 收集批次中的所有图和标签
        valid_count = 0
        for media_id in media_session_ids:
            graph = self.load_subgraph(media_id)
            labels = self.get_labels(media_id)

            if graph is not None and labels is not None:
                batch_graphs.append(graph)
                batch_labels.append(labels)
                valid_count += 1

        if valid_count > 0:
            # 合并图数据
            merged_data = self._merge_graphs(batch_graphs)
            if merged_data is None:
                return None

            # 添加标签到合并数据中
            merged_data['labels'] = torch.stack(batch_labels)

            return merged_data

        return None

    def _merge_graphs(self, graphs: List[HeteroData]) -> Dict:
        """合并多个异构图
        Args:
            graphs: HeteroData对象列表
        Returns:
            Dict: 包含合并后的节点特征和边索引的字典
        """
        if not graphs:
            return None

        # 初始化合并后的数据结构
        merged_data = {
            'x_dict': {},
            'edge_index_dict': {},
            'batch_dict': {},  # 添加批次索引字典
            'batch_size': len(graphs)
        }

        # 获取所有节点类型和边类型
        node_types = set()
        edge_types = set()
        for graph in graphs:
            node_types.update(graph.node_types)
            edge_types.update(graph.edge_types)

        # 记录每种节点类型的累积数量，用于边索引的偏移
        cumsum = {node_type: 0 for node_type in node_types}

        # 合并节点特征并创建批次索引
        for node_type in node_types:
            features = []
            batch_indices = []  # 用于记录每个节点属于哪个子图

            for graph_idx, graph in enumerate(graphs):
                if node_type in graph.node_types and hasattr(graph[node_type], 'x'):
                    if graph[node_type].x.shape[0] > 0:  # 只添加非空特征
                        num_nodes = graph[node_type].x.shape[0]
                        features.append(graph[node_type].x)
                        batch_indices.extend([graph_idx] * num_nodes)  # 为每个节点添加批次索引
                        cumsum[node_type] += num_nodes
                    else:
                        # 如果特征为空，创建一个具有正确维度的零张量
                        feature_dim = self._get_feature_dim(node_type)
                        empty_feature = torch.zeros((1, feature_dim), device=self.device)
                        features.append(empty_feature)
                        batch_indices.append(graph_idx)  # 为空节点添加批次索引
                        cumsum[node_type] += 1

            if features:
                merged_data['x_dict'][node_type] = torch.cat(features, dim=0)
                merged_data['batch_dict'][node_type] = torch.tensor(batch_indices, dtype=torch.long, device=self.device)
            else:
                # 如果没有这种类型的节点，创建一个空张量
                feature_dim = self._get_feature_dim(node_type)
                merged_data['x_dict'][node_type] = torch.zeros((1, feature_dim), device=self.device)
                merged_data['batch_dict'][node_type] = torch.zeros(1, dtype=torch.long, device=self.device)

        # 合并边索引
        offset = {node_type: 0 for node_type in node_types}
        for edge_type in edge_types:
            edge_indices = []

            for graph in graphs:
                if edge_type in graph.edge_types:
                    # 处理边索引
                    edge_index = graph[edge_type].edge_index.clone()
                    if edge_index.shape[1] > 0:  # 只处理非空边
                        # 更新源节点和目标节点的索引
                        edge_index[0] += offset[edge_type[0]]
                        edge_index[1] += offset[edge_type[-1]]
                        edge_indices.append(edge_index)

                # 更新偏移量
                if edge_type[0] in graph and hasattr(graph[edge_type[0]], 'x'):
                    offset[edge_type[0]] += max(1, graph[edge_type[0]].x.shape[0])
                if edge_type[-1] in graph and hasattr(graph[edge_type[-1]], 'x'):
                    offset[edge_type[-1]] += max(1, graph[edge_type[-1]].x.shape[0])

            if edge_indices:
                merged_data['edge_index_dict'][edge_type] = torch.cat(edge_indices, dim=1)
            else:
                merged_data['edge_index_dict'][edge_type] = torch.zeros((2, 0), dtype=torch.long, device=self.device)

        return merged_data

    def _get_feature_dim(self, node_type: str) -> int:
        """获取节点类型的特征维度"""
        dims = {
            'user': 34,           # 5(基本特征) + 29(扩展特征)
            'media_session': 781, # 10(原始特征) + 768(RoBERTa特征) + 3(VADER情感分数)
            'comment': 773,       # 2(原始特征: offensive, confidence) + 768(RoBERTa特征) + 3(VADER情感分数)
        }
        return dims.get(node_type, 768)