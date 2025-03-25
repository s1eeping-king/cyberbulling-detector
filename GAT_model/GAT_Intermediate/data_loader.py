import torch
from torch_geometric.data import HeteroData
from neo4j import GraphDatabase
from typing import Dict, List, Tuple

class DataLoader:
    """从Neo4j加载数据并处理为PyG格式的数据加载器"""
    
    def __init__(self, uri: str, user: str, password: str, device: torch.device):
        """初始化数据加载器"""
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.device = device
            
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
                return None
                
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
            return torch.zeros((0, 10), dtype=torch.float, device=self.device)
            
        properties = dict(media)
        # 将emotion和theme转换为数值特征
        emotion_map = {
            'neutral': 0.0,
            'joy': 1.0,
            'sad': -1.0,
            'love': 2.0,
            'surprise': 3.0,
            'fear': -2.0,
            'anger': -3.0
        }
        theme_map = {
            'people': 1.0,
            'person': 2.0,
            'indoor': 3.0,
            'outdoor': 4.0,
            'cartoon': 5.0,
            'text': 6.0,
            'activity': 7.0,
            'animal': 8.0,
            'other': 0.0
        }
        
        # 简单的文本编码
        description = str(properties.get('description', ''))
        desc_encoding = self._simple_text_encoding(description)
        
        features = [
            float(properties.get('likeCount', 0)),
            float(properties.get('commentCount', 0)),
            float(properties.get('loopCount', 0)),
            float(properties.get('repostCount', 0)),
            desc_encoding,  # 使用编码后的描述
            float(properties.get('emotion_confidence', 0)),
            float(properties.get('theme_confidence', 0)),
            float(properties.get('created', '0').replace('T', ' ').replace('Z', '').count(':')),
            emotion_map.get(properties.get('emotion', 'neutral'), 0.0),
            theme_map.get(properties.get('theme', 'other'), 0.0)
        ]
        
        return torch.tensor([features], dtype=torch.float, device=self.device)
        
    def _process_comment_features(self, comments) -> torch.Tensor:
        """处理评论节点特征"""
        if not comments:
            return torch.zeros((0, 2), dtype=torch.float, device=self.device)
            
        features_list = []
        for comment in comments:
            properties = dict(comment)
            # 简单的文本编码
            text = str(properties.get('text', ''))
            text_encoding = self._simple_text_encoding(text)
            
            features = [
                text_encoding,  # 使用编码后的文本
                float(hash(str(properties.get('postId', ''))) % 1000)
            ]
            features_list.append(features)
            
        return torch.tensor(features_list, dtype=torch.float, device=self.device)
        
    def _process_user_features(self, users) -> torch.Tensor:
        """处理用户节点特征"""
        if not users:
            return torch.zeros((0, 6), dtype=torch.float, device=self.device)
            
        # 使用字典进行去重，以用户ID为键
        unique_users = {}
        for user in users:
            properties = dict(user)
            user_id = str(properties.get('id', ''))
            if user_id not in unique_users:
                # 简单的文本编码
                username = str(properties.get('username', ''))
                description = str(properties.get('description', ''))
                username_encoding = self._simple_text_encoding(username)
                desc_encoding = self._simple_text_encoding(description)
                
                features = [
                    float(properties.get('followerCount', 0)),
                    float(properties.get('followingCount', 0)),
                    float(properties.get('likeCount', 0)),
                    float(properties.get('postCount', 0)),
                    username_encoding,  # 使用编码后的用户名
                    desc_encoding      # 使用编码后的描述
                ]
                unique_users[user_id] = features
            
        # 将去重后的特征转换为tensor
        features_list = list(unique_users.values())
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
            
        unique_nodes = set()
        valid_edges = []
        
        for e in edge_data:
            if e[0] is not None and e[1] is not None:
                unique_nodes.add(str(e[0]))
                unique_nodes.add(str(e[1]))
                valid_edges.append((str(e[0]), str(e[1])))
                
        if not valid_edges:
            return torch.zeros((2, 0), dtype=torch.long, device=self.device)
            
        id_to_idx = {node_id: idx for idx, node_id in enumerate(sorted(unique_nodes))}
        
        edge_indices = []
        for src, dst in valid_edges:
            src_idx = id_to_idx[src]
            dst_idx = id_to_idx[dst]
            edge_indices.append([src_idx, dst_idx])
            
        return torch.tensor(edge_indices, dtype=torch.long, device=self.device).t()
        
    def _get_bullying_labels(self, labels) -> torch.Tensor:
        """获取霸凌和攻击性标签"""
        if not labels:
            return torch.zeros((1, 2), dtype=torch.float, device=self.device)
            
        properties = dict(labels[0])
        bullying = float(properties.get('bullying', 'noneBll') == 'bullying')
        aggression = float(properties.get('aggression', 'noneAgg') == 'aggression')
            
        return torch.tensor([[bullying, aggression]], dtype=torch.float, device=self.device)
        
    def _simple_text_encoding(self, text: str) -> float:
        """简单的文本编码函数"""
        if not text:
            return 0.0
        # 将文本转换为数值：将每个字符的ASCII码相加并取平均
        return sum(ord(c) for c in text) / len(text)
        
    def close(self):
        """关闭数据库连接"""
        self.driver.close() 