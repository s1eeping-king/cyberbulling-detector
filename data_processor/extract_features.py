import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any
import os

class FeatureExtractor:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.nodes = {
            "comments": [],
            "labels": [],  # 只保留霸凌和攻击性标签
            "media_sessions": [],
            "users": []
        }
        self.relationships = {
            "publishes": [],      # user publish media
            "creates": [],        # user create comment
            "mentions": [],       # comment mention user
            "annotates": [],      # media has Label (只用于霸凌和攻击性标签)
            "belongs_to": []      # comment belongs to media session
        }

    def load_json_file(self, file_path: str) -> Dict:
        """Load a JSON file that may contain multiple JSON objects"""
        print(f"Loading JSON file from: {file_path}")
        result = {}
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            try:
                # Try to load as a single JSON object first
                data = json.loads(content)
                if isinstance(data, dict):
                    return data
                elif isinstance(data, list):
                    # If it's a list, convert to dict using _id as key
                    return {item['_id']: item for item in data if '_id' in item}
            except json.JSONDecodeError:
                # If that fails, try line by line
                result = {}
                for line in content.split('\n'):
                    line = line.strip()
                    if line:
                        try:
                            data = json.loads(line)
                            if '_id' in data:
                                result[data['_id']] = data
                        except json.JSONDecodeError as e:
                            print(f"Error decoding line: {e}")
                            print(f"Problematic line: {line[:200]}...")
                            continue
        return result

    def load_data(self):
        """Load all data sources"""
        try:
            # Load video data
            video_path = os.path.join(self.data_dir, "SampledASONAMPosts.json")
            self.video_data = self.load_json_file(video_path)
        
            # Load comment data
            comment_path = os.path.join(self.data_dir, "sampled_post-comments_vine.json")
            self.comment_data = self.load_json_file(comment_path)
        
            # Load user data
            user_path = os.path.join(self.data_dir, "vine_users_data.json")
            self.user_data = self.load_json_file(user_path)
        
            # Load URL to postID mapping
            url_mapping_path = os.path.join(self.data_dir, "urls_to_postids.txt")
            self.url_to_postid = {}
            with open(url_mapping_path, 'r', encoding='utf-8') as f:
                # Skip header line
                next(f)
                for line in f:
                    line = line.strip()
                    if not line:  # Skip empty lines
                        continue
                    postid, url = line.split(',')
                    self.url_to_postid[url] = postid
                    # Add mapping for URL with /embed/simple
                    if not url.endswith('/embed/simple'):
                        self.url_to_postid[url + '/embed/simple'] = postid
        
            # Load label data
            emotion_path = os.path.join(self.data_dir, "label_data/aggregate video emotion survey.csv")
            cyberbullying_path = os.path.join(self.data_dir, "label_data/vine_labeled_cyberbullying_data.csv")
            
            print(f"Loading emotion data from: {emotion_path}")
            self.emotion_data = pd.read_csv(emotion_path)
            
            print(f"Loading cyberbullying data from: {cyberbullying_path}")
            self.cyberbullying_data = pd.read_csv(cyberbullying_path)
            
            # Load preprocessed mention relationships
            mention_path = os.path.join(self.data_dir, "processed/mention_relationships.json")
            with open(mention_path, 'r', encoding='utf-8') as f:
                self.mention_relationships = json.load(f)
            
        except FileNotFoundError as e:
            print(f"Error: Could not find file: {e}")
            raise
        except Exception as e:
            print(f"Unexpected error while loading data: {e}")
            raise

    def extract_users(self):
        """Extract user nodes"""
        for user_id, user_info in self.user_data.items():
            user_node = {
                "id": user_id,
                "type": "user",
                "properties": {
                    "username": user_info.get("username", ""),
                    "description": user_info.get("description", ""),
                    "followerCount": user_info.get("followerCount", 0),
                    "followingCount": user_info.get("followingCount", 0),
                    "likeCount": user_info.get("likeCount", 0),
                    "postCount": user_info.get("postCount", 0)
                }
            }
            self.nodes["users"].append(user_node)

    def extract_media_sessions(self):
        """Extract media session nodes with emotion and topic labels as properties"""
        # 创建一个字典来存储每个视频的标签信息
        video_labels = {}
        
        # 处理情感标注数据
        for _, row in self.emotion_data.iterrows():
            video_url = row['videolink']
            post_id = self.url_to_postid.get(video_url)
            if not post_id:
                print(f"Warning: No postID found for URL: {video_url}")
                continue

            video_labels[post_id] = {
                "emotion": row.get("question2"),
                "theme": row.get("question3"),
                "emotion_confidence": row.get("question2:confidence"),
                "theme_confidence": row.get("question3:confidence")
            }

        # 处理网络欺凌标注数据
        for _, row in self.cyberbullying_data.iterrows():
            video_url = row['videolink']
            post_id = self.url_to_postid.get(video_url)
            if not post_id:
                print(f"Warning: No postID found for URL: {video_url}")
                continue

            if post_id in video_labels:
                video_labels[post_id].update({
                    "aggression": row.get("question1"),
                    "bullying": row.get("question2"),
                    "aggression_confidence": row.get("question1:confidence"),
                    "bullying_confidence": row.get("question2:confidence")
                })
            else:
                video_labels[post_id] = {
                    "aggression": row.get("question1"),
                    "bullying": row.get("question2"),
                    "aggression_confidence": row.get("question1:confidence"),
                    "bullying_confidence": row.get("question2:confidence")
                }

        # 创建媒体会话节点
        for post_id, post_info in self.video_data.items():
            video_url = post_info.get('permalinkUrl', '')
            
            # 获取该视频的标签信息
            labels = video_labels.get(post_id, {})
            
            media_session = {
                "id": post_id,
                "type": "media_session",
                "properties": {
                    "permalinkUrl": video_url,
                    "description": post_info.get("description", ""),
                    "likeCount": post_info.get("likeCount", 0),
                    "commentCount": post_info.get("commentCount", 0),
                    "loopCount": post_info.get("loopCount", 0),
                    "repostCount": post_info.get("repostCount", 0),
                    "created": post_info.get("created", ""),
                    "userId": post_info.get("userId", ""),
                    "username": post_info.get("username", ""),
                    # 添加情感和主题标签作为属性
                    "emotion": labels.get("emotion"),
                    "theme": labels.get("theme"),
                    "emotion_confidence": labels.get("emotion_confidence"),
                    "theme_confidence": labels.get("theme_confidence")
                }
            }
            self.nodes["media_sessions"].append(media_session)

    def extract_comments(self):
        """Extract comment nodes"""
        for full_id, post_data in self.comment_data.items():
            # 从完整ID中分离出postID和commentID
            try:
                post_id, comment_id = full_id.split('_')
            except ValueError:
                print(f"Warning: Invalid ID format: {full_id}")
                continue

            comment_text = post_data.get("commentText", "")
            if not comment_text or not isinstance(comment_text, str):
                continue
                
            comment_node = {
                "id": comment_id,  # 只使用commentID作为节点ID
                "type": "comment",
                "properties": {
                    "text": comment_text,
                    "postId": post_id,
                    "commentId": comment_id,
                    "userId": post_data.get("userId", "")  # 添加userId以便后续创建关系
                }
            }
            self.nodes["comments"].append(comment_node)

    def extract_labels(self):
        """只提取霸凌和攻击性标签节点"""
        for _, row in self.cyberbullying_data.iterrows():
            video_url = row['videolink']
            post_id = self.url_to_postid.get(video_url)
            if not post_id:
                print(f"Warning: No postID found for URL: {video_url}")
                continue

            # 只创建霸凌和攻击性标签节点
            label_node = {
                "id": f"label_{post_id}",
                "type": "label",
                "properties": {
                    "videolink": video_url,
                    "postId": post_id,
                    "aggression": row.get("question1"),
                    "bullying": row.get("question2"),
                    "aggression_confidence": row.get("question1:confidence"),
                    "bullying_confidence": row.get("question2:confidence")
                }
            }
            self.nodes["labels"].append(label_node)

    def extract_relationships(self):
        """Extract all relationships"""
        # 1. User publishes media relationship
        for post_id, post_info in self.video_data.items():
            user_id = post_info.get("userId", "")
            if user_id:
                publish_rel = {
                    "source": user_id,
                    "target": post_id,
                    "type": "publishes",
                    "properties": {
                        "timestamp": post_info.get("created", "")
                    }
                }
                self.relationships["publishes"].append(publish_rel)
        
        # 2. User creates comment relationship
        for comment_id, comment_data in self.comment_data.items():
            try:
                post_id, comment_id = comment_id.split('_')
            except ValueError:
                continue
                
            # Add creates relationship between user and comment
            user_id = comment_data.get("userId", "")
            if user_id:
                create_rel = {
                    "source": user_id,
                    "target": comment_id,
                    "type": "creates",
                    "properties": {}
                }
                self.relationships["creates"].append(create_rel)
            
            # Add belongs_to relationship between comment and media session
            belongs_to_rel = {
                "source": comment_id,
                "target": post_id,
                "type": "belongs_to",
                "properties": {}
            }
            self.relationships["belongs_to"].append(belongs_to_rel)
        
        # 3. Comment mentions user relationship (using preprocessed data)
        for mention in self.mention_relationships:
            mention_rel = {
                "source": mention["comment_id"],
                "target": mention["target_user_id"],
                "type": "mentions",
                "properties": {}
            }
            self.relationships["mentions"].append(mention_rel)
        
        # 4. Media has Label relationship (只用于霸凌和攻击性标签)
        for label in self.nodes["labels"]:
            post_id = label["properties"].get("postId")
            if post_id:
                annotate_rel = {
                    "source": post_id,
                    "target": label["id"],
                    "type": "annotates",
                    "properties": {}
                }
                self.relationships["annotates"].append(annotate_rel)

    def save_to_json(self, output_file: str = "data/processed/combined_features.json"):
        """Save all extracted features to a JSON file"""
        combined_data = {
            "nodes": {
                "comments": self.nodes["comments"],
                "labels": self.nodes["labels"],
                "media_sessions": self.nodes["media_sessions"],
                "users": self.nodes["users"]
            },
            "relationships": {
                "publishes": self.relationships["publishes"],
                "creates": self.relationships["creates"],
                "mentions": self.relationships["mentions"],
                "annotates": self.relationships["annotates"],
                "belongs_to": self.relationships["belongs_to"]
            }
        }
        
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(combined_data, f, indent=2, ensure_ascii=False)

    def extract_all(self):
        """Run all extraction methods"""
        self.load_data()
        self.extract_users()
        self.extract_media_sessions()
        self.extract_comments()
        self.extract_labels()
        self.extract_relationships()
        self.save_to_json()

if __name__ == "__main__":
    extractor = FeatureExtractor()
    extractor.extract_all()
