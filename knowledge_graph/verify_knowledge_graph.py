import pickle
import logging
import os
from typing import Dict, List, Optional
import pandas as pd
from data_loader import DataLoader

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def verify_knowledge_graph(graph_path: str = 'knowledge_graph/knowledge_graph.pkl'):
    """
    验证知识图谱中的节点信息，特别是检查视频节点的comment_count是否正确
    """
    try:
        # 加载知识图谱
        logger.info(f"Loading knowledge graph from {graph_path}")
        with open(graph_path, 'rb') as f:
            G = pickle.load(f)
        
        # 加载原始数据
        logger.info("Loading original data for comparison")
        data_loader = DataLoader()
        data_loader.load_data()
        
        # 验证视频节点的comment_count
        logger.info("Verifying video nodes comment_count...")
        video_nodes = [node for node in G.nodes(data=True) if node[1].get('node_type') == 'Video']
        
        # 打印前三个视频节点的信息
        logger.info("\nSample video nodes information:")
        for i, (node_id, node_data) in enumerate(video_nodes[:3]):
            video_id = node_data.get('id')
            kg_comment_count = node_data.get('comment_count', 0)
            
            # 获取原始数据中的评论数量
            actual_comments = data_loader.get_comments_for_video(video_id)
            actual_comment_count = len(actual_comments)
            
            logger.info(f"\nVideo Node {i+1}: {node_id}")
            logger.info(f"  - ID: {video_id}")
            logger.info(f"  - KG Comment Count: {kg_comment_count}")
            logger.info(f"  - Actual Comment Count: {actual_comment_count}")
            logger.info(f"  - Ratio: {kg_comment_count / actual_comment_count if actual_comment_count > 0 else 'N/A'}")
            
            # 打印前几条评论信息
            if actual_comments:
                logger.info(f"  - Sample Comments:")
                for j, comment in enumerate(actual_comments[:2]):
                    logger.info(f"    {j+1}. ID: {comment.get('_id')}, Text: {comment.get('commentText', '')[:50]}...")
        
        # 检查所有视频节点的comment_count
        count_mismatches = 0
        double_counts = 0
        total_videos = len(video_nodes)
        
        for node_id, node_data in video_nodes:
            video_id = node_data.get('id')
            kg_comment_count = node_data.get('comment_count', 0)
            actual_comment_count = len(data_loader.get_comments_for_video(video_id))
            
            if kg_comment_count != actual_comment_count:
                count_mismatches += 1
                
                # 检查是否是两倍关系
                if abs(kg_comment_count - 2 * actual_comment_count) < 2:  # 允许有小误差
                    double_counts += 1
        
        logger.info(f"\nComment Count Verification Summary:")
        logger.info(f"  - Total video nodes checked: {total_videos}")
        logger.info(f"  - Nodes with mismatched comment counts: {count_mismatches} ({count_mismatches/total_videos:.2%})")
        logger.info(f"  - Nodes with doubled comment counts: {double_counts} ({double_counts/total_videos:.2%})")
        
        # 检查媒体会话节点
        logger.info("\nVerifying media session nodes...")
        media_session_nodes = [node for node in G.nodes(data=True) if node[1].get('node_type') == 'MediaSession']
        
        # 打印前三个媒体会话节点的信息
        logger.info("\nSample media session nodes information:")
        for i, (node_id, node_data) in enumerate(media_session_nodes[:3]):
            video_id = node_data.get('video_id')
            session_comment_count = node_data.get('comment_count', 0)
            
            # 获取原始数据中的评论数量
            actual_comment_count = len(data_loader.get_comments_for_video(video_id))
            
            logger.info(f"\nMedia Session Node {i+1}: {node_id}")
            logger.info(f"  - Video ID: {video_id}")
            logger.info(f"  - Session Comment Count: {session_comment_count}")
            logger.info(f"  - Actual Comment Count: {actual_comment_count}")
            
        # 分析可能的问题原因
        logger.info("\nPossible causes of comment count issues:")
        logger.info("1. In kg_builder.py, comment_count is calculated twice:")
        logger.info("   - Once in build_video_nodes() using len(self.get_comments_for_video(video_id))")
        logger.info("   - Again in build_media_session_nodes() when creating media session nodes")
        logger.info("2. The get_comments_for_video() method might be counting comments differently than expected")
        logger.info("3. There might be duplicate comments in the original data")
        
        # 检查是否有重复评论
        logger.info("\nChecking for duplicate comments in the original data...")
        comment_ids = [comment.get('_id') for comment in data_loader.comment_data]
        unique_comment_ids = set(comment_ids)
        duplicate_count = len(comment_ids) - len(unique_comment_ids)
        
        logger.info(f"  - Total comments: {len(comment_ids)}")
        logger.info(f"  - Unique comment IDs: {len(unique_comment_ids)}")
        logger.info(f"  - Duplicate comments: {duplicate_count} ({duplicate_count/len(comment_ids):.2%} if any)")
        
        return G
        
    except Exception as e:
        logger.error(f"Error verifying knowledge graph: {str(e)}")
        raise

if __name__ == "__main__":
    verify_knowledge_graph('knowledge_graph/knowledge_graph.pkl')