import json
import logging
from typing import Dict, List

# def get_comments_for_video(video_id: str) -> List[Dict]:
#     """获取视频的所有评论"""
#     return get_comments_for_video1(video_id)

# def get_comments_for_video1(video_id: str) -> List[Dict]:
#     """获取视频的所有评论"""
#     return [comment for comment in comment_data if comment.get('postId') == video_id]

logger = logging.getLogger(__name__)
comment_data = []
    # 加载评论数据
video_id = "1100096249568595968"
with open('sampled_post-comments_vine.json', 'r', encoding='utf-8') as f:
    for line in f:
        try:
            comment = json.loads(line.strip())
            comment_data.append(comment)
        except json.JSONDecodeError as e:
            logger.warning(f"Error parsing comment data line: {e}")
            continue
comment = [comment for comment in comment_data if comment.get('postId') == video_id]
print(len(comment))