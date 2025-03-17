import pickle
from neo4j import GraphDatabase
import logging
from tqdm import tqdm
import torch
import numpy as np
from typing import List, Dict, Any
import math

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Neo4j连接信息
URI = "neo4j+s://696f4187.databases.neo4j.io"
USERNAME = "neo4j"
PASSWORD = "syhzZyADus__htxcihjQyVB1vOkEjpnS28yZtr_oKFg"

# 批处理大小
BATCH_SIZE = 100  # 减小批处理大小以降低内存使用

def convert_tensor_to_list(value):
    """将tensor转换为Python原生类型"""
    if isinstance(value, torch.Tensor):
        # 将tensor转换为numpy，再转换为list
        return value.detach().cpu().numpy().tolist()
    elif isinstance(value, np.ndarray):
        return value.tolist()
    elif isinstance(value, dict):
        return {k: convert_tensor_to_list(v) for k, v in value.items()}
    elif isinstance(value, (list, tuple)):
        return [convert_tensor_to_list(item) for item in value]
    return value

def flatten_dict(d, parent_key='', sep='_'):
    """将嵌套字典展平为单层字典"""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def filter_properties(properties: Dict[str, Any]) -> Dict[str, Any]:
    """过滤属性，只保留Neo4j支持的类型"""
    filtered = {}
    for k, v in properties.items():
        if isinstance(v, (str, int, float, bool)):
            filtered[k] = v
        elif isinstance(v, (list, tuple)) and len(v) > 0:
            # 只保留基本类型的列表
            if all(isinstance(x, (str, int, float, bool)) for x in v):
                filtered[k] = list(v)
    return filtered

class Neo4jUploader:
    def __init__(self, uri, username, password):
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
        
    def close(self):
        self.driver.close()
        
    def clear_database(self):
        """清空数据库"""
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            logger.info("Database cleared")
    
    def create_constraints(self):
        """创建约束以提高性能"""
        with self.driver.session() as session:
            # 为不同类型的节点创建约束
            constraints = [
                "CREATE CONSTRAINT IF NOT EXISTS FOR (v:Video) REQUIRE v.id IS UNIQUE",
                "CREATE CONSTRAINT IF NOT EXISTS FOR (u:User) REQUIRE u.id IS UNIQUE",
                "CREATE CONSTRAINT IF NOT EXISTS FOR (m:MediaSession) REQUIRE m.id IS UNIQUE",
                "CREATE CONSTRAINT IF NOT EXISTS FOR (l:Label) REQUIRE l.id IS UNIQUE"
            ]
            
            for constraint in constraints:
                try:
                    session.run(constraint)
                except Exception as e:
                    logger.warning(f"Constraint creation failed: {str(e)}")
            
            logger.info("Constraints created")
    
    def create_nodes_batch(self, tx, nodes_data):
        """批量创建节点"""
        for node_type, nodes in nodes_data.items():
            # 构建参数化查询
            query = (
                f"UNWIND $batch as row "
                f"MERGE (n:{node_type} {{id: row.id}}) "
                f"SET n = row.properties"
            )
            
            # 分批处理以避免内存问题
            for i in range(0, len(nodes), BATCH_SIZE):
                batch = nodes[i:i + BATCH_SIZE]
                try:
                    tx.run(query, batch=batch)
                except Exception as e:
                    logger.error(f"Error creating {node_type} nodes: {str(e)}")
                    raise
    
    def create_relationships_batch(self, tx, rels_data):
        """批量创建关系"""
        for rel_type, rels in rels_data.items():
            # 构建参数化查询
            query = (
                f"UNWIND $batch as row "
                f"MATCH (a {{id: row.start_id}}), (b {{id: row.end_id}}) "
                f"CREATE (a)-[r:{rel_type}]->(b) "
                f"SET r = row.properties"
            )
            
            # 分批处理以避免内存问题
            for i in range(0, len(rels), BATCH_SIZE):
                batch = rels[i:i + BATCH_SIZE]
                try:
                    tx.run(query, batch=batch)
                except Exception as e:
                    logger.error(f"Error creating {rel_type} relationships: {str(e)}")
                    raise
    
    def process_node_batch(self, nodes_batch):
        """处理一批节点数据"""
        nodes_by_type = {}
        
        for node_id, node_data in nodes_batch:
            node_type = node_data.get('node_type', 'Unknown')
            if node_type not in nodes_by_type:
                nodes_by_type[node_type] = []
            
            # 提取和处理属性
            properties = {k: v for k, v in node_data.items() if k != 'node_type'}
            
            # 处理特征向量
            if 'frame_features' in properties:
                features = properties['frame_features']
                if isinstance(features, torch.Tensor):
                    features = features.detach().cpu().numpy()
                elif isinstance(features, np.ndarray):
                    features = features
                
                # 计算统计特征
                properties['features_mean'] = float(np.mean(features))
                properties['features_std'] = float(np.std(features))
                properties['features_min'] = float(np.min(features))
                properties['features_max'] = float(np.max(features))
                del properties['frame_features']
            
            # 转换和过滤属性
            converted_properties = convert_tensor_to_list(properties)
            flattened_properties = flatten_dict(converted_properties)
            filtered_properties = filter_properties(flattened_properties)
            
            # 添加到对应类型的列表中
            nodes_by_type[node_type].append({
                'id': node_id,
                'properties': filtered_properties
            })
        
        return nodes_by_type
    
    def process_relationship_batch(self, rels_batch):
        """处理一批关系数据"""
        rels_by_type = {}
        
        for start_id, end_id, edge_data in rels_batch:
            rel_type = edge_data.get('relation', 'UNKNOWN')
            if rel_type not in rels_by_type:
                rels_by_type[rel_type] = []
            
            # 提取和处理属性
            properties = {k: v for k, v in edge_data.items() if k != 'relation'}
            
            # 转换和过滤属性
            converted_properties = convert_tensor_to_list(properties)
            flattened_properties = flatten_dict(converted_properties)
            filtered_properties = filter_properties(flattened_properties)
            
            # 添加到对应类型的列表中
            rels_by_type[rel_type].append({
                'start_id': start_id,
                'end_id': end_id,
                'properties': filtered_properties
            })
        
        return rels_by_type
    
    def upload_graph(self, graph):
        """上传整个图到Neo4j"""
        try:
            # 清空数据库
            self.clear_database()
            
            # 创建约束
            self.create_constraints()
            
            # 上传节点
            logger.info("Uploading nodes...")
            nodes_data = list(graph.nodes(data=True))
            total_nodes = len(nodes_data)
            
            with tqdm(total=total_nodes, desc="Processing nodes") as pbar:
                for i in range(0, total_nodes, BATCH_SIZE):
                    batch = nodes_data[i:i + BATCH_SIZE]
                    processed_batch = self.process_node_batch(batch)
                    
                    with self.driver.session() as session:
                        session.execute_write(self.create_nodes_batch, processed_batch)
                    
                    pbar.update(len(batch))
            
            # 上传边
            logger.info("Uploading relationships...")
            edges_data = list(graph.edges(data=True))
            total_edges = len(edges_data)
            
            with tqdm(total=total_edges, desc="Processing relationships") as pbar:
                for i in range(0, total_edges, BATCH_SIZE):
                    batch = edges_data[i:i + BATCH_SIZE]
                    processed_batch = self.process_relationship_batch(batch)
                    
                    with self.driver.session() as session:
                        session.execute_write(self.create_relationships_batch, processed_batch)
                    
                    pbar.update(len(batch))
            
            logger.info("Graph upload completed successfully")
            
        except Exception as e:
            logger.error(f"Error uploading graph: {str(e)}")
            raise

def main():
    try:
        # 读取知识图谱
        logger.info("Loading knowledge graph from pickle file...")
        with open('knowledge_graph.pkl', 'rb') as f:
            graph = pickle.load(f)
        
        # 创建上传器
        uploader = Neo4jUploader(URI, USERNAME, PASSWORD)
        
        # 上传图
        logger.info("Starting graph upload...")
        uploader.upload_graph(graph)
        
        # 关闭连接
        uploader.close()
        
        logger.info("Process completed successfully")
        
    except Exception as e:
        logger.error(f"Error in main process: {str(e)}")
        raise

if __name__ == "__main__":
    main()