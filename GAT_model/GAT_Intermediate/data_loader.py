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
            if self.debug:
                print(f"\n处理批次样本数: {valid_count}/{len(media_session_ids)}")
            
            # 合并图数据
            merged_data = self._merge_graphs(batch_graphs)
            if merged_data is None:
                return None
                
            # 添加标签到合并数据中
            merged_data['labels'] = torch.stack(batch_labels)
            
            if self.debug:
                # 打印每种节点类型的数量
                for node_type, x in merged_data['x_dict'].items():
                    print(f"{node_type} 节点数量: {x.shape[0]}")
                # 打印每种边类型的数量
                for edge_type, edge_index in merged_data['edge_index_dict'].items():
                    print(f"{edge_type} 边数量: {edge_index.shape[1]}")
                print(f"标签形状: {merged_data['labels'].shape}")
            
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
        
        # 合并节点特征
        for node_type in node_types:
            features = []
            for graph in graphs:
                if node_type in graph.node_types and hasattr(graph[node_type], 'x'):
                    if graph[node_type].x.shape[0] > 0:  # 只添加非空特征
                        features.append(graph[node_type].x)
                        cumsum[node_type] += graph[node_type].x.shape[0]
                    else:
                        # 如果特征为空，创建一个具有正确维度的零张量
                        feature_dim = self._get_feature_dim(node_type)
                        empty_feature = torch.zeros((1, feature_dim), device=self.device)
                        features.append(empty_feature)
                        cumsum[node_type] += 1
            
            if features:
                merged_data['x_dict'][node_type] = torch.cat(features, dim=0)
                if self.debug:
                    print(f"合并后 {node_type} 节点特征形状: {merged_data['x_dict'][node_type].shape}")
            else:
                # 如果没有这种类型的节点，创建一个空张量
                feature_dim = self._get_feature_dim(node_type)
                merged_data['x_dict'][node_type] = torch.zeros((1, feature_dim), device=self.device)
                if self.debug:
                    print(f"创建空 {node_type} 节点特征形状: {merged_data['x_dict'][node_type].shape}")
        
        # 合并边索引
        offset = {node_type: 0 for node_type in node_types}
        for edge_type in edge_types:
            edge_indices = []
            for graph in graphs:
                if edge_type in graph.edge_types:
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
                if self.debug:
                    print(f"合并后 {edge_type} 边索引形状: {merged_data['edge_index_dict'][edge_type].shape}")
            else:
                merged_data['edge_index_dict'][edge_type] = torch.zeros((2, 0), dtype=torch.long, device=self.device)
                if self.debug:
                    print(f"创建空 {edge_type} 边索引")
        
        return merged_data

    def _get_feature_dim(self, node_type: str) -> int:
        """获取节点类型的特征维度"""
        dims = {
            'user': 1540,        # 768(bert_username) + 768(bert_description) + 4(other_features)
            'media_session': 777,  # 768(bert_description) + 9(other_features)
            'comment': 769       # 768(bert_text) + 1(postId)
        }
        return dims.get(node_type, 64)