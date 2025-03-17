import pickle
import numpy as np
from typing import Tuple

def extract_features_and_labels(G) -> Tuple[np.ndarray, np.ndarray]:
    """从知识图谱中提取特征和标签"""
    features_list = []
    labels_list = []
    node_mapping = {}  # 用于映射节点ID到索引
    current_idx = 0

    # 遍历所有媒体会话节点
    for node in G.nodes():
        if node.startswith('media_session_'):
            node_data = G.nodes[node]
            video_id = node_data.get('video_id')
            
            # 获取视频节点的特征
            video_node = f"video_{video_id}"
            if video_node not in G:
                continue
            
            # 获取标签
            bullying_label = None
            confidence_score = 0.0  # 标签置信度
            for _, label_node in G.out_edges(node):
                label_data = G.nodes[label_node]
                if (label_data.get('node_type') == 'Label' and 
                    label_data.get('label_type') == 'bullying'):
                    bullying_label = label_data.get('value')
                    confidence_score = label_data.get('confidence', 1.0)  # 获取置信度，默认为1.0
                    break
            
            if bullying_label is not None:
                # 合并特征
                video_features = G.nodes[video_node].get('features_mean', 0)
                session_features = node_data.get('features_mean', 0)
                combined_features = np.array([
                    video_features,
                    session_features,
                    G.nodes[video_node].get('features_std', 0),
                    node_data.get('features_std', 0),
                    G.nodes[video_node].get('features_min', 0),
                    node_data.get('features_min', 0),
                    G.nodes[video_node].get('features_max', 0),
                    node_data.get('features_max', 0),
                ])
                
                features_list.append(combined_features)
                labels_list.append(bullying_label)
                node_mapping[node] = current_idx
                current_idx += 1

    return np.array(features_list), np.array(labels_list), node_mapping

with open('knowledge_graph.pkl', 'rb') as f:
    G = pickle.load(f)
features, labels, node_mapping = extract_features_and_labels(G)
print(features.shape, labels.shape)
# print(node_mapping)
