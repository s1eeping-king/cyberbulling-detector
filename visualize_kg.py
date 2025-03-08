import networkx as nx
import matplotlib.pyplot as plt
import pickle
import logging
from pathlib import Path
import os
import torch
import numpy as np
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from collections import Counter

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KnowledgeGraphVisualizer:
    def __init__(self, graph_path='knowledge_graph.pkl'):
        self.graph_path = graph_path
        self.output_dir = 'visualizations'
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 检查CUDA是否可用
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # 定义节点类型和颜色
        self.node_types = {
            'Video': '#FF9999',          # 红色
            'Comment': '#99CCFF',        # 蓝色
            'User': '#FF99CC',           # 粉色
            'AggressionLabel': '#FF6666',# 深红色
            'BullyingLabel': '#FF6666',  # 深红色
        }
        
        # 定义边类型和颜色
        self.edge_types = {
            'HAS_COMMENTS': '#0000FF',    # 蓝色
            'COMMENTED': '#00FFFF',       # 青色
            'HAS_OVERALL_LABEL': '#FFFF00',# 黄色
            'POSTED': '#008000',          # 深绿色
        }
    
    def load_graph(self):
        """加载知识图谱"""
        try:
            with open(self.graph_path, 'rb') as f:
                self.G = pickle.load(f)
            logger.info(f"Loaded knowledge graph with {self.G.number_of_nodes()} nodes and {self.G.number_of_edges()} edges")
            
            # 检查节点属性
            for node, data in self.G.nodes(data=True):
                if 'node_type' not in data:
                    logger.warning(f"Node {node} missing node_type attribute")
                    data['node_type'] = 'Unknown'  # 设置默认类型
            
            # 统计节点类型
            node_type_counts = Counter([data.get('node_type', 'Unknown') for _, data in self.G.nodes(data=True)])
            logger.info(f"Node type distribution: {node_type_counts}")
            
            # 统计边类型
            edge_type_counts = Counter([data.get('relation', 'Unknown') for _, _, data in self.G.edges(data=True)])
            logger.info(f"Edge type distribution: {edge_type_counts}")
            
            return True
        except Exception as e:
            logger.error(f"Error loading knowledge graph: {e}")
            return False
    
    def create_visualization_subgraph(self):
        """创建一个包含关键节点和关系的子图，适合可视化"""
        if not hasattr(self, 'G'):
            if not self.load_graph():
                return None
        
        # 选择一些视频节点作为起点
        video_nodes = [node for node, data in self.G.nodes(data=True) 
                      if data.get('node_type', 'Unknown') == 'Video']
        
        if not video_nodes:
            logger.warning("No video nodes found in the graph")
            return None
        
        # 选择少量视频节点
        if len(video_nodes) > 5:
            selected_videos = np.random.choice(video_nodes, 5, replace=False)
        else:
            selected_videos = video_nodes
        
        # 构建子图
        nodes_to_include = set(selected_videos)
        
        # 为每个视频添加相关节点
        for video in selected_videos:
            # 添加评论
            for neighbor in self.G.neighbors(video):
                if self.G.nodes[neighbor].get('node_type') == 'Comment':
                    nodes_to_include.add(neighbor)
                    
                    # 添加评论的用户
                    for user_neighbor in self.G.neighbors(neighbor):
                        if self.G.nodes[user_neighbor].get('node_type') == 'User':
                            nodes_to_include.add(user_neighbor)
            
            # 添加标签
            for neighbor in self.G.neighbors(video):
                if self.G.nodes[neighbor].get('node_type') in ['AggressionLabel', 'BullyingLabel']:
                    nodes_to_include.add(neighbor)
        
        # 创建子图
        subgraph = self.G.subgraph(nodes_to_include)
        logger.info(f"Created visualization subgraph with {subgraph.number_of_nodes()} nodes and {subgraph.number_of_edges()} edges")
        
        return subgraph
    
    def visualize_knowledge_graph(self, figsize=(20, 20)):
        """可视化知识图谱的关键部分"""
        subgraph = self.create_visualization_subgraph()
        if not subgraph:
            return
        
        output_path = os.path.join(self.output_dir, 'knowledge_graph.png')
        
        plt.figure(figsize=figsize)
        
        # 使用力导向布局
        logger.info("Computing layout...")
        pos = nx.spring_layout(subgraph, k=0.5, iterations=100, seed=42)
        
        # 按节点类型绘制节点
        node_sizes = {
            'Video': 2500,
            'Comment': 1500,
            'User': 2000,
            'AggressionLabel': 1800,
            'BullyingLabel': 1800,
            'Unknown': 1000
        }
        
        for node_type, color in self.node_types.items():
            nodes = [node for node, data in subgraph.nodes(data=True) 
                    if data.get('node_type', 'Unknown') == node_type]
            if nodes:
                nx.draw_networkx_nodes(subgraph, pos, 
                                     nodelist=nodes,
                                     node_color=color,
                                     node_size=node_sizes.get(node_type, 1000),
                                     alpha=0.8,
                                     label=node_type)
        
        # 按边类型绘制边
        for edge_type, color in self.edge_types.items():
            edges = [(u, v) for u, v, data in subgraph.edges(data=True) 
                    if data.get('relation') == edge_type]
            if edges:
                nx.draw_networkx_edges(subgraph, pos,
                                     edgelist=edges,
                                     edge_color=color,
                                     width=2,
                                     alpha=0.7,
                                     label=edge_type)
        
        # 添加标签
        labels = {}
        for node in subgraph.nodes():
            node_type = subgraph.nodes[node].get('node_type', 'Unknown')
            if node_type == 'Video':
                # 视频节点显示简短标签
                labels[node] = f"Video: {str(node)[:15]}..." if len(str(node)) > 15 else f"Video: {node}"
            elif node_type == 'User':
                # 用户节点显示简短标签
                labels[node] = f"User: {str(node)[:15]}..." if len(str(node)) > 15 else f"User: {node}"
            elif node_type == 'Comment':
                # 评论节点显示简短内容
                labels[node] = f"Comment: {str(node)[:10]}..." if len(str(node)) > 10 else f"Comment: {node}"
            elif node_type in ['AggressionLabel', 'BullyingLabel']:
                # 标签节点显示完整标签
                labels[node] = f"{node_type}: {node}"
            else:
                # 其他节点显示类型和简短标签
                labels[node] = f"{node_type}: {str(node)[:10]}..." if len(str(node)) > 10 else f"{node_type}: {node}"
        
        nx.draw_networkx_labels(subgraph, pos, labels, font_size=10, font_weight='bold')
        
        # 添加图例
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 添加标题
        plt.title("Knowledge Graph Visualization\n(Showing Key Relationships Between Videos, Comments, Users and Labels)", 
                 pad=20, fontsize=16)
        
        # 保存图片
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()
        logger.info(f"Knowledge graph visualization saved to {output_path}")
    
    def visualize_video_comment_network(self, figsize=(20, 20)):
        """可视化视频-评论-用户网络"""
        if not hasattr(self, 'G'):
            if not self.load_graph():
                return
        
        output_path = os.path.join(self.output_dir, 'video_comment_network.png')
        
        # 选择视频节点
        video_nodes = [node for node, data in self.G.nodes(data=True) 
                      if data.get('node_type', 'Unknown') == 'Video']
        
        if len(video_nodes) > 10:
            selected_videos = np.random.choice(video_nodes, 10, replace=False)
        else:
            selected_videos = video_nodes
        
        # 构建子图
        nodes_to_include = set(selected_videos)
        
        # 为每个视频添加评论和用户
        for video in selected_videos:
            for neighbor in self.G.neighbors(video):
                if self.G.nodes[neighbor].get('node_type') == 'Comment':
                    nodes_to_include.add(neighbor)
                    
                    # 添加评论的用户
                    for user_neighbor in self.G.neighbors(neighbor):
                        if self.G.nodes[user_neighbor].get('node_type') == 'User':
                            nodes_to_include.add(user_neighbor)
        
        # 创建子图
        subgraph = self.G.subgraph(nodes_to_include)
        
        plt.figure(figsize=figsize)
        
        # 使用二分图布局
        logger.info("Computing bipartite layout...")
        
        # 分离不同类型的节点
        videos = [n for n, d in subgraph.nodes(data=True) if d.get('node_type') == 'Video']
        comments = [n for n, d in subgraph.nodes(data=True) if d.get('node_type') == 'Comment']
        users = [n for n, d in subgraph.nodes(data=True) if d.get('node_type') == 'User']
        
        # 创建分层布局
        pos = {}
        
        # 视频在左侧
        for i, node in enumerate(videos):
            pos[node] = (0, i * 10 / (len(videos) + 1))
        
        # 评论在中间
        for i, node in enumerate(comments):
            pos[node] = (0.5, i * 10 / (len(comments) + 1))
        
        # 用户在右侧
        for i, node in enumerate(users):
            pos[node] = (1, i * 10 / (len(users) + 1))
        
        # 绘制节点
        nx.draw_networkx_nodes(subgraph, pos, 
                             nodelist=videos,
                             node_color=self.node_types['Video'],
                             node_size=2500,
                             alpha=0.8,
                             label='Video')
        
        nx.draw_networkx_nodes(subgraph, pos, 
                             nodelist=comments,
                             node_color=self.node_types['Comment'],
                             node_size=1500,
                             alpha=0.8,
                             label='Comment')
        
        nx.draw_networkx_nodes(subgraph, pos, 
                             nodelist=users,
                             node_color=self.node_types['User'],
                             node_size=2000,
                             alpha=0.8,
                             label='User')
        
        # 绘制边
        video_comment_edges = [(u, v) for u, v, d in subgraph.edges(data=True) 
                              if (subgraph.nodes[u].get('node_type') == 'Video' and 
                                  subgraph.nodes[v].get('node_type') == 'Comment') or
                                 (subgraph.nodes[u].get('node_type') == 'Comment' and 
                                  subgraph.nodes[v].get('node_type') == 'Video')]
        
        comment_user_edges = [(u, v) for u, v, d in subgraph.edges(data=True) 
                             if (subgraph.nodes[u].get('node_type') == 'Comment' and 
                                 subgraph.nodes[v].get('node_type') == 'User') or
                                (subgraph.nodes[u].get('node_type') == 'User' and 
                                 subgraph.nodes[v].get('node_type') == 'Comment')]
        
        nx.draw_networkx_edges(subgraph, pos,
                             edgelist=video_comment_edges,
                             edge_color='blue',
                             width=1.5,
                             alpha=0.6,
                             label='Video-Comment')
        
        nx.draw_networkx_edges(subgraph, pos,
                             edgelist=comment_user_edges,
                             edge_color='green',
                             width=1.5,
                             alpha=0.6,
                             label='Comment-User')
        
        # 添加标签
        video_labels = {node: f"V: {str(node)[:10]}..." for node in videos}
        comment_labels = {node: f"C: {str(node)[:10]}..." for node in comments}
        user_labels = {node: f"U: {str(node)[:10]}..." for node in users}
        
        labels = {**video_labels, **comment_labels, **user_labels}
        nx.draw_networkx_labels(subgraph, pos, labels, font_size=8)
        
        # 添加图例
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 添加标题
        plt.title("Video-Comment-User Network", pad=20, fontsize=16)
        
        # 保存图片
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()
        logger.info(f"Video-comment network visualization saved to {output_path}")
    
    def visualize_label_distribution(self, figsize=(12, 8)):
        """可视化标签分布"""
        if not hasattr(self, 'G'):
            if not self.load_graph():
                return
        
        output_path = os.path.join(self.output_dir, 'label_distribution.png')
        
        # 获取所有视频节点
        video_nodes = [node for node, data in self.G.nodes(data=True) 
                      if data.get('node_type', 'Unknown') == 'Video']
        
        # 统计标签分布
        aggression_labels = []
        bullying_labels = []
        
        for video in video_nodes:
            for neighbor in self.G.neighbors(video):
                node_type = self.G.nodes[neighbor].get('node_type', 'Unknown')
                if node_type == 'AggressionLabel':
                    aggression_labels.append(neighbor)
                elif node_type == 'BullyingLabel':
                    bullying_labels.append(neighbor)
        
        # 统计标签值
        aggression_counts = Counter(aggression_labels)
        bullying_counts = Counter(bullying_labels)
        
        plt.figure(figsize=figsize)
        
        # 创建子图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # 绘制Aggression标签分布
        if aggression_counts:
            labels, values = zip(*sorted(aggression_counts.items()))
            ax1.bar(labels, values, color='red', alpha=0.7)
            ax1.set_title('Aggression Label Distribution')
            ax1.set_ylabel('Count')
            ax1.tick_params(axis='x', rotation=45)
        else:
            ax1.text(0.5, 0.5, 'No Aggression Labels Found', 
                    horizontalalignment='center', verticalalignment='center')
        
        # 绘制Bullying标签分布
        if bullying_counts:
            labels, values = zip(*sorted(bullying_counts.items()))
            ax2.bar(labels, values, color='blue', alpha=0.7)
            ax2.set_title('Bullying Label Distribution')
            ax2.set_ylabel('Count')
            ax2.tick_params(axis='x', rotation=45)
        else:
            ax2.text(0.5, 0.5, 'No Bullying Labels Found', 
                    horizontalalignment='center', verticalalignment='center')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
        logger.info(f"Label distribution visualization saved to {output_path}")

def main():
    # 创建可视化器
    visualizer = KnowledgeGraphVisualizer()
    
    # 可视化知识图谱
    visualizer.visualize_knowledge_graph(figsize=(25, 20))
    
    # 可视化视频-评论-用户网络
    visualizer.visualize_video_comment_network(figsize=(20, 15))
    
    # 可视化标签分布
    visualizer.visualize_label_distribution(figsize=(15, 10))

if __name__ == "__main__":
    main() 