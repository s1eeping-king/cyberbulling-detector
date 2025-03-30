import torch
from torch_geometric.data import HeteroData
from neo4j import GraphDatabase
from typing import Dict, List, Tuple
from transformers import DistilBertTokenizer, DistilBertModel

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
            
            # 3. 处理边索引，使用已建立的用户ID映射
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
        """使用BERT获取文本嵌入"""
        if not text:
            return torch.zeros(768, device=self.device)
            
        # 对文本进行编码
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # 获取BERT输出
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
            # 使用[CLS]token的输出作为文本表示
            embeddings = outputs.last_hidden_state[:, 0, :].squeeze(0)
            
        return embeddings
            
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
        
        # 获取描述文本的BERT嵌入
        description = str(properties.get('description', ''))
        desc_embedding = self._get_bert_embedding(description)
        
        # 其他特征
        other_features = torch.tensor([
            float(properties.get('likeCount', 0)),
            float(properties.get('commentCount', 0)),
            float(properties.get('loopCount', 0)),
            float(properties.get('repostCount', 0)),
            float(properties.get('emotion_confidence', 0)),
            float(properties.get('theme_confidence', 0)),
            float(properties.get('created', '0').replace('T', ' ').replace('Z', '').count(':')),
            emotion_map.get(properties.get('emotion', 'neutral'), 0.0),
            theme_map.get(properties.get('theme', 'other'), 0.0)
        ], device=self.device)
        
        # 合并特征并确保是二维张量
        combined_features = torch.cat([desc_embedding, other_features])
        # 如果是一维张量，转换为二维
        if combined_features.dim() == 1:
            combined_features = combined_features.unsqueeze(0)
        return combined_features
        
    def _process_comment_features(self, comments) -> torch.Tensor:
        """处理评论节点特征"""
        if not comments:
            # 返回一个占位符特征，而不是空张量，维度为(1, 769)，与预期的评论特征维度匹配
            # 768(bert_text) + 1(postId)
            return torch.zeros((1, 769), dtype=torch.float, device=self.device)
            
        features_list = []
        for comment in comments:
            properties = dict(comment)
            # 获取评论文本的BERT嵌入
            text = str(properties.get('text', ''))
            text_embedding = self._get_bert_embedding(text)
            
            # 其他特征
            other_features = torch.tensor([
                float(hash(str(properties.get('postId', ''))) % 1000)
            ], device=self.device)
            
            # 合并特征
            features = torch.cat([text_embedding, other_features])
            features_list.append(features)
            
        return torch.stack(features_list)
        
    def _process_user_features(self, users) -> Tuple[torch.Tensor, Dict[str, int]]:
        """处理用户节点特征，返回特征矩阵和ID到索引的映射"""
        if not users:
            return torch.zeros((0, 1540), dtype=torch.float, device=self.device), {}
            
        # 使用字典进行去重，以用户elementId为键
        unique_users = {}
        id_to_idx = {}
        current_idx = 0
        
        for user in users:
            properties = dict(user)
            # 修改这里：从获取id改为获取elementId
            user_id = str(user.element_id)  # 使用节点的element_id属性
            if user_id not in unique_users:
                # 获取用户名和描述的BERT嵌入
                username = str(properties.get('username', ''))
                description = str(properties.get('description', ''))
                username_embedding = self._get_bert_embedding(username)
                desc_embedding = self._get_bert_embedding(description)
                
                # 其他特征
                other_features = torch.tensor([
                    float(properties.get('followerCount', 0)),
                    float(properties.get('followingCount', 0)),
                    float(properties.get('likeCount', 0)),
                    float(properties.get('postCount', 0))
                ], device=self.device)
                
                # 合并特征
                features = torch.cat([username_embedding, desc_embedding, other_features])
                unique_users[user_id] = features
                id_to_idx[user_id] = current_idx
                current_idx += 1
            
        # 将去重后的特征转换为tensor
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