import torch
from torch_geometric.data import HeteroData
from neo4j import GraphDatabase
from typing import Dict, List, Tuple
import json
import os
import numpy as np

class DataLoader:
    """从Neo4j加载数据并处理为PyG格式的数据加载器"""

    def __init__(self, uri: str, user: str, password: str, device: torch.device, debug: bool = False):
        """初始化数据加载器"""
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.device = device
        self.debug = debug  # 调试模式标志
        self.debug_counter = 0  # 调试计数器

        # 加载BERT特征
        self.comment_bert_features = self._load_bert_features("data/processed/bert_features/comment_bert_features.json", "comment")
        self.media_bert_features = self._load_bert_features("data/processed/bert_features/media_bert_features.json", "media")

        print(f"已加载 {len(self.comment_bert_features)} 条评论BERT特征")
        print(f"已加载 {len(self.media_bert_features)} 个媒体会话BERT特征")

    def _load_bert_features(self, file_path: str, feature_type: str) -> Dict[str, torch.Tensor]:
        """加载BERT特征

        Args:
            file_path: BERT特征文件路径
            feature_type: 特征类型 ('comment' 或 'media')

        Returns:
            Dict[str, torch.Tensor]: ID到特征的映射
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
            features_dict[id_] = torch.tensor(feature, dtype=torch.float)

        return features_dict

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
        """处理媒体会话节点特征，整合BERT特征"""
        if not media:
            # 返回一个占位符特征，维度为(1, 10+64)，包括原始特征和BERT特征
            return torch.zeros((1, 74), dtype=torch.float, device=self.device)

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

        # 获取BERT特征
        if media_id in self.media_bert_features:
            bert_features = self.media_bert_features[media_id].to(self.device)
        else:
            # 如果没有找到BERT特征，使用零向量
            bert_features = torch.zeros(64, device=self.device)

        # 合并原始特征和BERT特征
        features = torch.cat([original_features, bert_features])

        # 确保是二维张量
        if features.dim() == 1:
            features = features.unsqueeze(0)
        return features

    def _process_comment_features(self, comments) -> Tuple[torch.Tensor, Dict[str, int]]:
        """处理评论节点特征，返回特征矩阵和ID到索引的映射"""
        if not comments:
            return torch.zeros((0, 66), dtype=torch.float, device=self.device), {}

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

                # BERT特征
                comment_db_id = properties.get('id', '')
                if comment_db_id in self.comment_bert_features:
                    bert_features = self.comment_bert_features[comment_db_id].to(self.device)
                else:
                    bert_features = torch.zeros(64, device=self.device)

                # 合并特征
                features = torch.cat([basic_features, bert_features])

                unique_comments[comment_id] = features
                id_to_idx[comment_id] = current_idx
                current_idx += 1

        # 将去重后的特征转换为tensor
        features_list = list(unique_comments.values())
        if not features_list:
            return torch.zeros((0, 66), dtype=torch.float, device=self.device), {}

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
            return torch.zeros((0, 5), dtype=torch.float, device=self.device), {}

        # 使用字典进行去重，以用户elementId为键
        unique_users = {}
        id_to_idx = {}
        current_idx = 0

        for user in users:
            properties = dict(user)
            # 修改这里：从获取id改为获取elementId
            user_id = str(user.element_id)  # 使用节点的element_id属性
            if user_id not in unique_users:
                # 处理字符串类型的属性
                description_offensive = properties.get('description_offensive', 'not_offensive')
                description_offensive_value = 1.0 if description_offensive == 'offensive' else 0.0

                # 直接获取已经处理好的属性
                features = torch.tensor([
                    description_offensive_value,
                    float(properties.get('followerCount_normalized', 0.0)),
                    float(properties.get('followingCount_normalized', 0.0)),
                    float(properties.get('likeCount_normalized', 0.0)),
                    float(properties.get('postCount_normalized', 0.0))
                ], device=self.device)

                unique_users[user_id] = features
                id_to_idx[user_id] = current_idx
                current_idx += 1

        # 将去重后的特征转换为tensor
        features_list = list(unique_users.values())
        return torch.stack(features_list), id_to_idx

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
            'user': 5,           # 5(description_offensive, followerCount_normalized, followingCount_normalized, likeCount_normalized, postCount_normalized)
            'media_session': 74, # 10(原始特征) + 64(BERT特征)
            'comment': 66,       # 2(原始特征: offensive, confidence) + 64(BERT特征)
        }
        return dims.get(node_type, 64)