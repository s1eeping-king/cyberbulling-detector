import torch
from torch_geometric.data import HeteroData, Batch
from neo4j import GraphDatabase
from typing import Dict, List, Tuple
from transformers import DistilBertTokenizer, DistilBertModel
from torch_geometric.loader import DataLoader as PyGDataLoader
import torch.nn as nn

class DataLoader:
    """从Neo4j加载数据并处理为PyG格式的数据加载器"""
    
    def __init__(self, uri: str, user: str, password: str, device: torch.device, debug: bool = False):
        """初始化数据加载器"""
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.device = device
        self.debug = debug  # 调试模式标志
        self.debug_counter = 0  # 调试计数器
        # 初始化DistilBERT
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.bert_model = DistilBertModel.from_pretrained('distilbert-base-uncased').to(device)
        self.bert_model.eval()  # 设置为评估模式
        
        # 添加BERT特征降维层
        self.bert_dim = 128  # 降维后的BERT特征维度
        self.bert_projection = nn.Linear(768, self.bert_dim).to(device)
            
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
        OPTIONAL MATCH (publisher:User)-[r5:PUBLISHES]->(m)
        OPTIONAL MATCH (creator:User)-[r2:CREATES]->(c)
        OPTIONAL MATCH (c)-[r3:MENTIONS]->(mentioned:User)
        
        RETURN m,
               collect(distinct c) as comments,
               collect(distinct creator) as comment_creators,
               collect(distinct publisher) as publishers,
               collect(distinct mentioned) as mentioned_users,
               collect(distinct [elementId(c), elementId(m)]) as belongs_to_rels,
               collect(distinct [elementId(creator), elementId(c)]) as creates_rels,
               collect(distinct [elementId(c), elementId(mentioned)]) as mentions_rels,
               collect(distinct [elementId(publisher), elementId(m)]) as publishes_rels
        """
        
        with self.driver.session() as session:
            result = session.run(query, media_id=media_session_id).single()
            if not result:
                return None
                
            data = HeteroData()
            
            # 1. 首先处理所有用户节点，建立ID到索引的映射
            all_users = result['comment_creators'] + result['publishers'] + result['mentioned_users']
            user_features, user_id_to_idx = self._process_user_features(all_users)
            data['user'].x = user_features
            
            # 2. 处理其他节点特征
            data['media_session'].x = self._process_media_features(result['m'])
            data['comment'].x = self._process_comment_features(result['comments'])
            
            # 3. 添加super节点 - 使用正确的维度
            super_dim = self._get_feature_dim('super')  # 现在会返回128
            data['super'].x = torch.zeros((1, super_dim), dtype=torch.float, device=self.device)
            
            # 4. 创建到super节点的单向边
            # user -> super
            num_users = data['user'].x.shape[0]
            if num_users > 0:
                data['user', 'to_super', 'super'].edge_index = torch.tensor([
                    list(range(num_users)),  # source nodes
                    [0] * num_users          # target node (super node)
                ], dtype=torch.long, device=self.device)
            
            # media_session -> super
            num_media = data['media_session'].x.shape[0]
            if num_media > 0:
                data['media_session', 'to_super', 'super'].edge_index = torch.tensor([
                    list(range(num_media)),
                    [0] * num_media
                ], dtype=torch.long, device=self.device)
            
            # comment -> super
            num_comments = data['comment'].x.shape[0]
            if num_comments > 0:
                data['comment', 'to_super', 'super'].edge_index = torch.tensor([
                    list(range(num_comments)),
                    [0] * num_comments
                ], dtype=torch.long, device=self.device)
            
            # 5. 处理原有的边索引
            data['comment', 'belongs_to', 'media_session'].edge_index = self._process_edge_index(
                result['belongs_to_rels'], 
                {'comment': data['comment'].x.shape[0], 'media_session': data['media_session'].x.shape[0]}
            )
            data['user', 'creates', 'comment'].edge_index = self._process_edge_index(
                result['creates_rels'],
                {'user': data['user'].x.shape[0], 'comment': data['comment'].x.shape[0]},
                user_id_to_idx
            )
            data['comment', 'mentions', 'user'].edge_index = self._process_edge_index(
                result['mentions_rels'],
                {'comment': data['comment'].x.shape[0], 'user': data['user'].x.shape[0]},
                user_id_to_idx
            )
            data['user', 'publishes', 'media_session'].edge_index = self._process_edge_index(
                result['publishes_rels'],
                {'user': data['user'].x.shape[0], 'media_session': data['media_session'].x.shape[0]},
                user_id_to_idx
            )
            
            return data
            
    def _get_bert_embedding(self, text: str) -> torch.Tensor:
        """使用BERT获取文本嵌入并降维"""
        if not text:
            return torch.zeros(self.bert_dim, device=self.device)
            
        # 对文本进行编码
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # 获取BERT输出并降维
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :].squeeze(0)  # [768]
            projected_embeddings = self.bert_projection(embeddings)  # [128]
            
        return projected_embeddings
            
    def _process_media_features(self, media) -> torch.Tensor:
        """处理媒体会话节点特征"""
        if not media:
            return torch.zeros((0, self.bert_dim + 9), dtype=torch.float, device=self.device)
            
        properties = dict(media)
        
        # BERT特征
        description = str(properties.get('description', ''))
        desc_embedding = self._get_bert_embedding(description)  # [128]
        
        # 数值特征（进行归一化）
        numerical_features = torch.tensor([
            float(properties.get('likeCount', 0)),
            float(properties.get('commentCount', 0)),
            float(properties.get('loopCount', 0)),
            float(properties.get('repostCount', 0)),
            float(properties.get('emotion_confidence', 0)),
            float(properties.get('theme_confidence', 0)),
            float(properties.get('created', '0').replace('T', ' ').replace('Z', '').count(':')),
            self._get_emotion_encoding(properties.get('emotion', 'neutral')),
            self._get_theme_encoding(properties.get('theme', 'other'))
        ], device=self.device)
        
        # 对数变换处理大数值
        numerical_features[:4] = torch.log1p(numerical_features[:4])
        
        # 合并特征
        combined_features = torch.cat([desc_embedding, numerical_features])
        return combined_features.unsqueeze(0) if combined_features.dim() == 1 else combined_features
        
    def _process_comment_features(self, comments) -> torch.Tensor:
        """处理评论节点特征"""
        if not comments:
            return torch.zeros((1, self.bert_dim + 1), dtype=torch.float, device=self.device)
            
        features_list = []
        for comment in comments:
            properties = dict(comment)
            # BERT特征
            text = str(properties.get('text', ''))
            text_embedding = self._get_bert_embedding(text)  # [128]
            
            # ID特征
            id_feature = torch.tensor([
                float(hash(str(properties.get('postId', ''))) % 1000)
            ], device=self.device)
            
            # 合并特征
            features = torch.cat([text_embedding, id_feature])
            features_list.append(features)
            
        return torch.stack(features_list)
        
    def _process_user_features(self, users) -> Tuple[torch.Tensor, Dict[str, int]]:
        """处理用户节点特征，返回特征矩阵和ID到索引的映射"""
        if not users:
            return torch.zeros((0, self.bert_dim * 2 + 4), dtype=torch.float, device=self.device), {}
            
        unique_users = {}
        id_to_idx = {}
        current_idx = 0
        
        for user in users:
            properties = dict(user)
            user_id = str(user.element_id)
            if user_id not in unique_users:
                # BERT特征
                username = str(properties.get('username', ''))
                description = str(properties.get('description', ''))
                username_embedding = self._get_bert_embedding(username)  # [128]
                desc_embedding = self._get_bert_embedding(description)   # [128]
                
                # 数值特征（进行归一化）
                numerical_features = torch.tensor([
                    float(properties.get('followerCount', 0)),
                    float(properties.get('followingCount', 0)),
                    float(properties.get('likeCount', 0)),
                    float(properties.get('postCount', 0))
                ], device=self.device)
                
                # 对数变换处理大数值
                numerical_features = torch.log1p(numerical_features)
                
                # 合并特征
                features = torch.cat([username_embedding, desc_embedding, numerical_features])
                unique_users[user_id] = features
                id_to_idx[user_id] = current_idx
                current_idx += 1
        
        features_list = list(unique_users.values())
        return torch.stack(features_list), id_to_idx
        
    def _process_edge_index(self, edge_data, node_counts, user_id_to_idx=None) -> torch.Tensor:
        """处理边索引，使用节点计数和用户ID映射"""
        if not edge_data:
            return torch.zeros((2, 0), dtype=torch.long, device=self.device)
            
        edge_indices = []
        for e in edge_data:
            if e[0] is not None and e[1] is not None:
                src_id = str(e[0])
                dst_id = str(e[1])
                
                # 处理源节点索引
                if user_id_to_idx is not None and src_id in user_id_to_idx:
                    src_idx = user_id_to_idx[src_id]
                else:
                    # 对于非用户节点，使用简单的哈希映射
                    src_idx = hash(src_id) % node_counts[list(node_counts.keys())[0]]
                
                # 处理目标节点索引
                if user_id_to_idx is not None and dst_id in user_id_to_idx:
                    dst_idx = user_id_to_idx[dst_id]
                else:
                    # 对于非用户节点，使用简单的哈希映射
                    dst_idx = hash(dst_id) % node_counts[list(node_counts.keys())[1]]
                
                # 验证索引是否有效
                if (src_idx < node_counts[list(node_counts.keys())[0]] and 
                    dst_idx < node_counts[list(node_counts.keys())[1]]):
                    edge_indices.append([src_idx, dst_idx])
                
        if not edge_indices:
            return torch.zeros((2, 0), dtype=torch.long, device=self.device)
            
        return torch.tensor(edge_indices, dtype=torch.long, device=self.device).t()
        
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
                return torch.zeros((2,), dtype=torch.float, device=self.device)
                
            labels = result['labels']
            if not labels:
                return torch.zeros((2,), dtype=torch.float, device=self.device)
                
            properties = dict(labels[0])
            bullying = float(properties.get('bullying', 'noneBll') == 'bullying')
            aggression = float(properties.get('aggression', 'noneAgg') == 'aggression')
                
            return torch.tensor([bullying, aggression], dtype=torch.float, device=self.device)
        
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
            # 使用PyG的Batch类合并图
            batch = Batch.from_data_list(batch_graphs)
            
            # 转换为所需的格式
            merged_data = {
                'x_dict': {},
                'edge_index_dict': {},
                'labels': torch.stack(batch_labels)
            }
            
            # 复制节点特征，为super节点创建独立特征
            for node_type in batch.node_types:
                if node_type == 'super':
                    # 为每个子图创建一个独立的super节点
                    feature_dim = self._get_feature_dim('super')
                    merged_data['x_dict']['super'] = torch.randn((len(batch_graphs), feature_dim), 
                                                               device=self.device) * 0.01
                else:
                    merged_data['x_dict'][node_type] = batch[node_type].x
            
            # 复制边索引，特别处理super节点的边
            for edge_type in batch.edge_types:
                if edge_type[-1] == 'super':
                    # 处理到super节点的边
                    src_nodes = batch[edge_type].edge_index[0]
                    batch_idx = batch[edge_type[0]].batch[src_nodes]  # 获取源节点所属的批次索引
                    edge_index = torch.stack([
                        src_nodes,
                        batch_idx  # 使用批次索引作为目标super节点的索引
                    ])
                    merged_data['edge_index_dict'][edge_type] = edge_index
                else:
                    merged_data['edge_index_dict'][edge_type] = batch[edge_type].edge_index
            
            return merged_data
        
        return None

    def _get_emotion_encoding(self, emotion: str) -> float:
        """获取情感的数值编码"""
        emotion_map = {
            'neutral': 0.0,
            'joy': 0.8,
            'love': 1.0,
            'surprise': 0.4,
            'sad': -0.6,
            'fear': -0.8,
            'anger': -1.0
        }
        return emotion_map.get(emotion, 0.0)
        
    def _get_theme_encoding(self, theme: str) -> float:
        """获取主题的数值编码"""
        theme_map = {
            'people': 0.2,
            'person': 0.3,
            'indoor': 0.4,
            'outdoor': 0.5,
            'cartoon': 0.6,
            'text': 0.7,
            'activity': 0.8,
            'animal': 0.9,
            'other': 0.1
        }
        return theme_map.get(theme, 0.1)
        
    def _get_feature_dim(self, node_type: str) -> int:
        """获取节点类型的特征维度"""
        dims = {
            'user': self.bert_dim * 2 + 4,        # 128(bert_username) + 128(bert_description) + 4(other_features)
            'media_session': self.bert_dim + 9,    # 128(bert_description) + 9(other_features)
            'comment': self.bert_dim + 1,         # 128(bert_text) + 1(postId)
            'super': 128                          # 与模型的hidden_dim保持一致
        }
        return dims.get(node_type, 128)