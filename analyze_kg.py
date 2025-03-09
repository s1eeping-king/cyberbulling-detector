from typing import Tuple
import pickle
import torch
import logging

logger = logging.getLogger(__name__)

def load_knowledge_graph(path: str = 'knowledge_graph.pkl') -> Tuple[torch.Tensor, torch.Tensor]:
    """加载知识图谱数据"""
    try:
        with open(path, 'rb') as f:
            G = pickle.load(f)
        logger.info(f"Successfully loaded knowledge graph from {path}")
        
        # 统计节点类型
        node_type_counts = {}
        for node, attr in G.nodes(data=True):
            node_type = attr.get('node_type', 'Unknown')
            node_type_counts[node_type] = node_type_counts.get(node_type, 0) + 1
        
        # 统计关系类型
        edge_type_counts = {}
        for _, _, attr in G.edges(data=True):
            edge_type = attr.get('relation', 'Unknown')
            edge_type_counts[edge_type] = edge_type_counts.get(edge_type, 0) + 1
        
        # 打印详细统计信息
        logger.info("\nDetailed Knowledge Graph Statistics:")
        logger.info("\nNode Types Distribution:")
        for node_type, count in sorted(node_type_counts.items()):
            logger.info(f"- {node_type}: {count:,} nodes ({count/G.number_of_nodes():.2%})")
        
        logger.info("\nEdge Types Distribution:")
        for edge_type, count in sorted(edge_type_counts.items()):
            logger.info(f"- {edge_type}: {count:,} edges ({count/G.number_of_edges():.2%})")
        
        # 获取节点特征和边索引
        num_nodes = G.number_of_nodes()
        nodes = list(G.nodes(data=True))
        
        # 确定特征维度
        feature_dim = 768  # 默认特征维度
        for node_id, node_data in nodes:
            if 'features' in node_data:
                if isinstance(node_data['features'], torch.Tensor):
                    feature_dim = node_data['features'].shape[-1]
                else:
                    feature_dim = len(node_data['features'])
                break
        
        # 初始化节点特征矩阵
        node_features = torch.zeros((num_nodes, feature_dim))
        node_id_map = {}  # 用于映射原始节点ID到连续索引
        
        # 构建节点特征矩阵和ID映射
        for idx, (node_id, node_data) in enumerate(nodes):
            node_id_map[node_id] = idx
            if 'features' in node_data:
                features = node_data['features']
                if isinstance(features, torch.Tensor):
                    # 如果是CUDA张量，先转到CPU
                    if features.is_cuda:
                        features = features.cpu()
                    # 确保是浮点类型
                    features = features.float()
                    # 如果是2D张量，取第一行
                    if features.dim() > 1:
                        features = features.view(-1)[:feature_dim]
                    node_features[idx] = features.clone().detach()
                else:
                    # 如果是其他类型（如numpy数组或列表），转换为张量
                    node_features[idx] = torch.tensor(features, dtype=torch.float32)
        
        # 构建边索引矩阵
        edge_index = []
        for src, dst in G.edges():
            edge_index.append([node_id_map[src], node_id_map[dst]])
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        
        # 打印基本统计信息
        logger.info(f"\nBasic Knowledge Graph Statistics:")
        logger.info(f"- Total number of nodes: {num_nodes:,}")
        logger.info(f"- Total number of edges: {len(G.edges()):,}")
        logger.info(f"- Average degree: {2*len(G.edges())/num_nodes:.2f}")
        logger.info(f"- Feature dimension: {feature_dim}")
        logger.info(f"- Node features shape: {node_features.shape}")
        logger.info(f"- Edge index shape: {edge_index.shape}")
        
        return node_features, edge_index
        
    except Exception as e:
        logger.error(f"Error loading knowledge graph: {str(e)}")
        raise

if __name__ == "__main__":
    # 设置日志级别
    logging.basicConfig(level=logging.INFO)
    
    try:
        # 加载并分析知识图谱
        logger.info("Loading and analyzing knowledge graph...")
        node_features, edge_index = load_knowledge_graph('knowledge_graph.pkl')
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise
