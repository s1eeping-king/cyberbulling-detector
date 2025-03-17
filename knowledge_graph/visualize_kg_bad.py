import networkx as nx
import plotly.graph_objects as go
import pickle
import random
import numpy as np
from collections import defaultdict
import logging
import matplotlib.colors as mcolors

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class KnowledgeGraphVisualizer:
    """知识图谱可视化器"""
    
    def __init__(self, graph_path='knowledge_graph/knowledge_graph.pkl'):
        """
        初始化可视化器
        Args:
            graph_path: 知识图谱pickle文件路径
        """
        self.load_graph(graph_path)
        
        # 节点类型的颜色映射
        self.node_colors = {
            'Video': '#1f77b4',      # 蓝色
            'User': '#2ca02c',       # 绿色
            'MediaSession': '#ff7f0e',# 橙色
            'Label': '#d62728'       # 红色
        }
        
        # 边类型的颜色映射
        self.edge_colors = {
            'USER_POSTED_VIDEO': '#aec7e8',    # 浅蓝
            'USER_COMMENTED_VIDEO': '#98df8a',  # 浅绿
            'VIDEO_HAS_SESSION': '#ffbb78',     # 浅橙
            'HAS_LABEL': '#ff9896'              # 浅红
        }
    
    def load_graph(self, graph_path):
        """加载知识图谱"""
        try:
            with open(graph_path, 'rb') as f:
                self.G = pickle.load(f)
            logger.info(f"Successfully loaded graph from {graph_path}")
            logger.info(f"Graph has {len(self.G.nodes)} nodes and {len(self.G.edges)} edges")
            
            # 打印节点类型分布
            node_types = defaultdict(int)
            for node, data in self.G.nodes(data=True):
                node_types[data.get('node_type', 'Unknown')] += 1
            logger.info("Node type distribution:")
            for ntype, count in node_types.items():
                logger.info(f"- {ntype}: {count}")
            
            # 打印边类型分布
            edge_types = defaultdict(int)
            for _, _, data in self.G.edges(data=True):
                edge_types[data.get('relation', 'Unknown')] += 1
            logger.info("Edge type distribution:")
            for etype, count in edge_types.items():
                logger.info(f"- {etype}: {count}")
                
        except Exception as e:
            logger.error(f"Error loading graph: {str(e)}")
            raise
    
    def get_node_info(self, node_id):
        """获取节点信息的字符串表示"""
        node_data = self.G.nodes[node_id]
        node_type = node_data.get('node_type', 'Unknown')
        
        info = f"ID: {node_id}<br>Type: {node_type}<br>"
        
        # 根据节点类型添加特定信息
        if node_type == 'Video':
            info += f"Likes: {node_data.get('likes_count', 'N/A')}<br>"
            info += f"Comments: {node_data.get('comment_count', 'N/A')}"
        elif node_type == 'User':
            info += f"Posts: {node_data.get('post_count', 'N/A')}<br>"
            info += f"Comments: {node_data.get('comment_count', 'N/A')}"
        elif node_type == 'Label':
            info += f"Label Type: {node_data.get('label_type', 'N/A')}<br>"
            info += f"Value: {node_data.get('value', 'N/A')}"
        elif node_type == 'MediaSession':
            info += f"Video ID: {node_data.get('video_id', 'N/A')}"
        
        return info
    
    def get_subgraph(self, num_nodes=100, seed_node=None):
        """
        获取以seed_node为中心的子图
        如果没有指定seed_node，则随机选择一个Video节点作为起点
        """
        # 如果没有指定seed_node，优先选择Video类型的节点
        if seed_node is None:
            video_nodes = [n for n, d in self.G.nodes(data=True) 
                         if d.get('node_type') == 'Video']
            if video_nodes:
                seed_node = random.choice(video_nodes)
            else:
                seed_node = random.choice(list(self.G.nodes()))
        
        # 使用BFS获取邻近节点
        subgraph_nodes = set()
        queue = [seed_node]
        while len(subgraph_nodes) < num_nodes and queue:
            current = queue.pop(0)
            if current not in subgraph_nodes:
                subgraph_nodes.add(current)
                # 优先添加与当前节点直接相连的节点
                neighbors = list(self.G.neighbors(current))
                random.shuffle(neighbors)  # 随机打乱邻居顺序
                queue.extend(neighbors)
        
        return self.G.subgraph(list(subgraph_nodes))
    
    def visualize_subgraph(self, num_nodes=100, seed_node=None, layout='spring', height=800):
        """
        可视化子图
        Args:
            num_nodes: 子图节点数量
            seed_node: 中心节点
            layout: 布局算法 ('spring' or 'circular')
            height: 图的高度
        """
        # 获取子图
        subgraph = self.get_subgraph(num_nodes, seed_node)
        nodes = list(subgraph.nodes())
        
        if not nodes:
            logger.error("No nodes in subgraph!")
            return None
        
        # 计算布局
        if layout == 'spring':
            pos = nx.spring_layout(subgraph, k=2/np.sqrt(len(nodes)), iterations=100)
        else:
            pos = nx.circular_layout(subgraph)
        
        # 准备边轨迹
        edge_traces = []
        for edge in subgraph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            relation = subgraph.edges[edge].get('relation', 'Unknown')
            color = self.edge_colors.get(relation, '#888')
            
            edge_trace = go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                line=dict(width=1, color=color),
                hoverinfo='none',
                mode='lines',
                showlegend=False
            )
            edge_traces.append(edge_trace)
        
        # 准备节点轨迹
        node_traces = []
        for node_type in self.node_colors:
            # 获取特定类型的节点
            type_nodes = [n for n in nodes if subgraph.nodes[n].get('node_type') == node_type]
            if not type_nodes:
                continue
                
            x_coords = [pos[node][0] for node in type_nodes]
            y_coords = [pos[node][1] for node in type_nodes]
            texts = [self.get_node_info(node) for node in type_nodes]
            
            node_trace = go.Scatter(
                x=x_coords,
                y=y_coords,
                mode='markers',
                hoverinfo='text',
                text=texts,
                name=f'Node: {node_type}',
                marker=dict(
                    size=20 if node_type == 'Video' else 15 if node_type == 'User' else 25 if node_type == 'MediaSession' else 10,
                    color=self.node_colors[node_type],
                    line=dict(width=1, color='#888')
                )
            )
            node_traces.append(node_trace)
        
        # 创建图形
        fig = go.Figure(
            data=edge_traces + node_traces,
            layout=go.Layout(
                title=f'Knowledge Graph Visualization (Showing {len(nodes)} nodes)',
                showlegend=True,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                annotations=[],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                height=height,
                width=height * 1.5
            )
        )
        
        # 添加边类型图例
        for relation, color in self.edge_colors.items():
            fig.add_trace(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode='lines',
                    name=f'Edge: {relation}',
                    line=dict(color=color, width=2),
                    showlegend=True
                )
            )
        
        return fig
    
    def save_visualization(self, fig, output_path='knowledge_graph/knowledge_graph_viz.html'):
        """保存可视化结果"""
        if fig is None:
            logger.error("No figure to save!")
            return
            
        try:
            fig.write_html(output_path)
            logger.info(f"Visualization saved to {output_path}")
        except Exception as e:
            logger.error(f"Error saving visualization: {str(e)}")
            raise

def main():
    # 创建可视化器
    visualizer = KnowledgeGraphVisualizer()
    
    # 生成并保存不同视图
    # 1. 默认视图（100个节点，以视频节点为中心）
    fig1 = visualizer.visualize_subgraph(num_nodes=100)
    if fig1:
        visualizer.save_visualization(fig1, 'kg_viz_100nodes.html')
    
    # 2. 较小的视图（50个节点）
    fig2 = visualizer.visualize_subgraph(num_nodes=50)
    if fig2:
        visualizer.save_visualization(fig2, 'kg_viz_50nodes.html')
    
    # 3. 圆形布局
    fig3 = visualizer.visualize_subgraph(num_nodes=100, layout='circular')
    if fig3:
        visualizer.save_visualization(fig3, 'kg_viz_circular.html')
    
    logger.info("Visualization process completed")

if __name__ == "__main__":
    main()
