import json
import logging
import os
import glob
import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Tuple, Optional, Set

# 设置日志
logger = logging.getLogger(__name__)

class DataLoader:
    """
    数据加载器类，负责加载所有必要的数据
    """
    def __init__(self, device=None):
        # 设置设备
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 初始化数据容器
        self.video_data = []
        self.comment_data = []
        self.user_data = []
        self.cyberbullying_data = None
        self.emotion_data = None
        self.individual_emotion_data = None
        self.frame_features = {}
        self.comment_features = None
        self.comment_ids = []
        
        # 映射关系
        self.postid_to_videolink = {}
        self.videolink_to_postid = {}
        self.valid_user_ids = set()
    
    def load_data(self):
        """加载所有必要的数据"""
        try:
            logger.info("Loading data...")
            
            # 加载视频数据
            with open('SampledASONAMPosts.json', 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        video = json.loads(line.strip())
                        self.video_data.append(video)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Error parsing video data line: {e}")
                        continue
            
            # 加载评论数据
            with open('sampled_post-comments_vine.json', 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        comment = json.loads(line.strip())
                        self.comment_data.append(comment)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Error parsing comment data line: {e}")
                        continue
            
            # 加载用户数据
            with open('vine_users_data.json', 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        user = json.loads(line.strip())
                        self.user_data.append(user)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Error parsing user data line: {e}")
                        continue
            
            # 创建用户ID集合
            self.valid_user_ids = {user['_id'] for user in self.user_data}
            
            # 加载标签数据
            self.cyberbullying_data = pd.read_csv('label_data/vine_labeled_cyberbullying_data.csv')
            self.emotion_data = pd.read_csv('label_data/aggregate video emotion survey.csv')
            self.individual_emotion_data = pd.read_csv('label_data/individual video emotion survey.csv')
            
            # 加载postId和videolink的映射关系
            # 尝试多个可能的路径
            possible_paths = [
                'urls_to_postids.txt',
                'label_data/urls_to_postids.txt',
                'processed/urls_to_postids.txt'
            ]
            
            mapping_loaded = False
            for path in possible_paths:
                try:
                    if os.path.exists(path):
                        logger.info(f"Loading mapping from {path}")
                        with open(path, 'r') as f:
                            for line in f:
                                parts = line.strip().split(',')
                                if len(parts) == 2:
                                    postid, videolink = parts
                                    self.postid_to_videolink[postid] = videolink
                                    self.videolink_to_postid[videolink] = postid
                        mapping_loaded = True
                        logger.info(f"Loaded {len(self.postid_to_videolink)} mappings from {path}")
                        break
                except Exception as e:
                    logger.warning(f"Error loading mapping from {path}: {e}")
            
            # 如果映射为空，创建一个基于数据的映射
            if not mapping_loaded or not self.videolink_to_postid:
                logger.warning("PostID to VideoLink mapping not loaded from file, creating from data")
                self._create_fallback_mapping()
            
            # 加载视频帧特征
            frame_features_dir = 'processed/frame_features'
            if os.path.exists(frame_features_dir):
                for npy_file in glob.glob(os.path.join(frame_features_dir, '*.npy')):
                    post_id = os.path.splitext(os.path.basename(npy_file))[0]
                    features = np.load(npy_file)
                    self.frame_features[post_id] = torch.from_numpy(features).float()
                logger.info(f"Loaded {len(self.frame_features)} frame feature files")
            
            # 加载评论特征
            try:
                comment_features_path = 'processed/vine_features.npy'
                if os.path.exists(comment_features_path):
                    self.comment_features = np.load(comment_features_path)
                    logger.info(f"Loaded comment features with shape: {self.comment_features.shape}")
                
                # 加载评论ID映射
                comment_ids_path = 'processed/comment_ids.txt'
                if os.path.exists(comment_ids_path):
                    with open(comment_ids_path, 'r') as f:
                        self.comment_ids = [line.strip() for line in f]
                    logger.info(f"Loaded {len(self.comment_ids)} comment IDs")
            except Exception as e:
                logger.warning(f"Error loading comment features: {e}")
            
            logger.info(f"Successfully loaded data:")
            logger.info(f"- Videos: {len(self.video_data)}")
            logger.info(f"- Comments: {len(self.comment_data)}")
            logger.info(f"- Users: {len(self.user_data)}")
            logger.info(f"- Valid user IDs: {len(self.valid_user_ids)}")
            logger.info(f"- PostID to VideoLink mappings: {len(self.postid_to_videolink)}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def _create_fallback_mapping(self):
        """创建备用的postId和videolink映射"""
        logger.info("Creating fallback mapping from data")
        
        # 从视频数据中提取postId和url
        for video in self.video_data:
            post_id = video.get('_id')
            url = video.get('url', '')
            
            if post_id and url:
                # 提取url的最后部分作为videolink
                videolink = url.split('/')[-1] if '/' in url else url
                self.postid_to_videolink[post_id] = videolink
                self.videolink_to_postid[videolink] = post_id
        
        # 从cyberbullying_data中提取更多映射
        if hasattr(self, 'cyberbullying_data') and self.cyberbullying_data is not None:
            # 直接从标签数据中提取videolink
            videolinks = set(self.cyberbullying_data['videolink'].unique())
            logger.info(f"Found {len(videolinks)} unique videolinks in cyberbullying data")
            
            # 尝试匹配视频URL
            for videolink in videolinks:
                if videolink not in self.videolink_to_postid:
                    for video in self.video_data:
                        post_id = video.get('_id')
                        url = video.get('url', '')
                        
                        # 如果url包含videolink，建立映射
                        if post_id and url and (videolink in url or url.endswith(videolink)):
                            self.postid_to_videolink[post_id] = videolink
                            self.videolink_to_postid[videolink] = post_id
                            break
        
        # 从frame_features目录中提取postId
        frame_features_dir = 'processed/frame_features'
        if os.path.exists(frame_features_dir):
            for npy_file in glob.glob(os.path.join(frame_features_dir, '*.npy')):
                post_id = os.path.splitext(os.path.basename(npy_file))[0]
                
                # 如果这个postId不在映射中，尝试从视频数据中找到对应的URL
                if post_id not in self.postid_to_videolink:
                    for video in self.video_data:
                        if video.get('_id') == post_id:
                            url = video.get('url', '')
                            if url:
                                videolink = url.split('/')[-1] if '/' in url else url
                                self.postid_to_videolink[post_id] = videolink
                                self.videolink_to_postid[videolink] = post_id
                                break
        
        logger.info(f"Created fallback mapping with {len(self.postid_to_videolink)} entries")
    
    def get_post_id_from_url(self, videolink: str) -> Optional[str]:
        """从视频链接获取postId"""
        return self.videolink_to_postid.get(videolink)
    
    def get_comments_for_video(self, video_id: str) -> List[Dict]:
        """获取视频的所有评论"""
        return [comment for comment in self.comment_data if comment.get('postId') == video_id]
    
    def get_comment_features(self, video_id: str) -> Optional[np.ndarray]:
        """获取视频评论的特征向量"""
        if self.comment_features is None or not self.comment_ids:
            return None
        
        # 找到该视频的所有评论ID
        video_comment_indices = []
        for i, comment_id in enumerate(self.comment_ids):
            if comment_id.startswith(f"{video_id}_"):
                video_comment_indices.append(i)
        
        if not video_comment_indices:
            return None
        
        # 提取并聚合特征
        comment_features = self.comment_features[video_comment_indices]
        return np.mean(comment_features, axis=0) if len(comment_features) > 0 else None


def load_all_data(device=None):
    """
    加载所有数据的便捷函数
    
    参数:
        device: 计算设备 (CPU/GPU)
    
    返回:
        data_loader: 加载了所有数据的DataLoader实例
    """
    data_loader = DataLoader(device)
    data_loader.load_data()
    return data_loader
