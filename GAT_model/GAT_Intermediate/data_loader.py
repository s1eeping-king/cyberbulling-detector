import torch
from torch_geometric.data import HeteroData
from neo4j import GraphDatabase
from typing import Dict, List, Tuple
import logging

class DataLoader:
    """从Neo4j加载数据并处理为PyG格式的数据加载器"""
    
    def __init__(self, uri: str, user: str, password: str, device: torch.device):
        """初始化数据加载器"""
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.device = device
        self.logger = self.setup_logger()
        
    @staticmethod
    def setup_logger():
        """设置日志"""
        logger = logging.getLogger('DataLoader')
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger
        
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
        """加载单个媒体会话的子图"""
        # 构建查询
        query = """
        MATCH (m:MediaSession {id: $media_id})
        OPTIONAL MATCH (m)<-[r1:BELONGS_TO]-(c:Comment)
        OPTIONAL MATCH (u:User)-[r2:CREATES]->(c)
        OPTIONAL MATCH (c)-[r3:MENTIONS]->(mentioned:User)
        OPTIONAL MATCH (m)-[r4:HAS_LABEL]->(l:Label)
        OPTIONAL MATCH (u)-[r5:PUBLISHES]->(m)
        RETURN m,
               collect(distinct c) as comments,
               collect(distinct u) as users,
               collect(distinct mentioned) as mentioned_users,
               collect(distinct l) as labels,
               collect(distinct [elementId(c), elementId(m)]) as belongs_to_rels,
               collect(distinct [elementId(u), elementId(c)]) as creates_rels,
               collect(distinct [elementId(c), elementId(mentioned)]) as mentions_rels,
               collect(distinct [elementId(m), elementId(l)]) as has_label_rels,
               collect(distinct [elementId(u), elementId(m)]) as publishes_rels
        """
        
        with self.driver.session() as session:
            result = session.run(query, media_id=media_session_id).single()
            if not result:
                self.logger.warning(f"No data found for media session {media_session_id}")
                return None
                
            # 打印原始查询结果中的标签信息
            # self.logger.info(f"Media session {media_session_id}:")
            # self.logger.info(f"Labels found: {len(result['labels'])}")
            # if result['labels']:
            #     self.logger.info(f"First label properties: {dict(result['labels'][0])}")
            # self.logger.info(f"Has_label relationships: {result['has_label_rels']}")
                
            # 创建异构图数据对象
            data = HeteroData()
            
            # 处理节点特征
            data['media_session'].x = self._process_media_features(result['m'])
            data['comment'].x = self._process_comment_features(result['comments'])
            data['user'].x = self._process_user_features(result['users'] + result['mentioned_users'])
            data['label'].x = self._process_label_features(result['labels'])
            
            # 处理边索引
            data['comment', 'belongs_to', 'media_session'].edge_index = self._process_edge_index(result['belongs_to_rels'])
            data['user', 'creates', 'comment'].edge_index = self._process_edge_index(result['creates_rels'])
            data['comment', 'mentions', 'user'].edge_index = self._process_edge_index(result['mentions_rels'])
            data['media_session', 'has_label', 'label'].edge_index = self._process_edge_index(result['has_label_rels'])
            data['user', 'publishes', 'media_session'].edge_index = self._process_edge_index(result['publishes_rels'])
            
            # 添加标签（用于训练）
            data['media_session'].y = self._get_bullying_labels(result['labels'])
            
            return data
            
    def _process_media_features(self, media) -> torch.Tensor:
        """处理媒体会话节点特征"""
        if not media:
            return torch.zeros((0, 8), dtype=torch.float, device=self.device)
            
        properties = dict(media)
        
        # 提取数值特征
        features = [
            float(properties.get('likeCount', 0)),
            float(properties.get('commentCount', 0)),
            float(properties.get('loopCount', 0)),
            float(properties.get('repostCount', 0)),
            # 文本长度特征
            len(str(properties.get('description', ''))),
            # 情感和主题的置信度
            float(properties.get('emotion_confidence', 0)),
            float(properties.get('theme_confidence', 0)),
            # 创建时间（转换为时间戳）
            float(properties.get('created', '0').replace('T', ' ').replace('Z', '').count(':'))  # 简单处理，仅用于示例
        ]
        
        return torch.tensor([features], dtype=torch.float, device=self.device)
        
    def _process_comment_features(self, comments) -> torch.Tensor:
        """处理评论节点特征"""
        if not comments:
            return torch.zeros((0, 2), dtype=torch.float, device=self.device)
            
        features_list = []
        for comment in comments:
            properties = dict(comment)
            features = [
                len(str(properties.get('text', ''))),
                float(hash(str(properties.get('postId', ''))) % 1000)  # 添加postId的哈希值作为特征
            ]
            features_list.append(features)
            
        return torch.tensor(features_list, dtype=torch.float, device=self.device)
        
    def _process_user_features(self, users) -> torch.Tensor:
        """处理用户节点特征"""
        if not users:
            return torch.zeros((0, 6), dtype=torch.float, device=self.device)
            
        features_list = []
        for user in users:
            properties = dict(user)
            features = [
                float(properties.get('followerCount', 0)),
                float(properties.get('followingCount', 0)),
                float(properties.get('likeCount', 0)),
                float(properties.get('postCount', 0)),
                len(str(properties.get('username', ''))),
                len(str(properties.get('description', '')))
            ]
            features_list.append(features)
            
        return torch.tensor(features_list, dtype=torch.float, device=self.device)
        
    def _process_label_features(self, labels) -> torch.Tensor:
        """处理标签节点特征"""
        if not labels:
            return torch.zeros((0, 4), dtype=torch.float, device=self.device)
            
        features_list = []
        for label in labels:
            properties = dict(label)
            features = [
                float(properties.get('bullying', 'noneBll') == 'bullying'),
                float(properties.get('aggression', 'noneAgg') == 'aggression'),
                float(properties.get('bullying_confidence', 0)),
                float(properties.get('aggression_confidence', 0))
            ]
            features_list.append(features)
            
        return torch.tensor(features_list, dtype=torch.float, device=self.device)
        
    def _process_edge_index(self, edge_data) -> torch.Tensor:
        """处理边索引"""
        if not edge_data:
            return torch.zeros((2, 0), dtype=torch.long, device=self.device)
            
        # 创建节点ID到索引的映射
        unique_nodes = set()
        valid_edges = []
        
        # 收集所有有效的边
        for e in edge_data:
            if e[0] is not None and e[1] is not None:
                unique_nodes.add(str(e[0]))
                unique_nodes.add(str(e[1]))
                valid_edges.append((str(e[0]), str(e[1])))
                
        if not valid_edges:
            return torch.zeros((2, 0), dtype=torch.long, device=self.device)
            
        # 创建连续的索引映射
        id_to_idx = {node_id: idx for idx, node_id in enumerate(sorted(unique_nodes))}
        
        # 转换为索引
        edge_indices = []
        for src, dst in valid_edges:
            src_idx = id_to_idx[src]
            dst_idx = id_to_idx[dst]
            edge_indices.append([src_idx, dst_idx])
            
        return torch.tensor(edge_indices, dtype=torch.long, device=self.device).t()
        
    def _get_bullying_labels(self, labels) -> torch.Tensor:
        """获取霸凌和攻击性标签"""
        if not labels:
            # self.logger.warning("No labels found for this media session")
            return torch.zeros((1, 2), dtype=torch.float, device=self.device)
            
        # 从第一个标签获取信息（因为每个媒体会话只有一个标签）
        properties = dict(labels[0])
        
        # 检查标签值
        bullying = float(properties.get('bullying', 'noneBll') == 'bullying')
        aggression = float(properties.get('aggression', 'noneAgg') == 'aggression')
        
        # 打印标签信息用于调试
        # if bullying == 1.0 or aggression == 1.0:
        #     self.logger.info(f"Found positive label - Bullying: {bullying} ({properties.get('bullying')}), "
        #                    f"Aggression: {aggression} ({properties.get('aggression')})")
            
        return torch.tensor([[bullying, aggression]], dtype=torch.float, device=self.device)
        
    def close(self):
        """关闭数据库连接"""
        self.driver.close() 