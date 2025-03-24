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
            "annotations": [],
            "media_sessions": [],
            "users": []
        }
        self.relationships = {
            "posts": [],
            "comments": [],
            "interactions": []
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
        
            # Load annotation data
            emotion_path = os.path.join(self.data_dir, "label_data/aggregate video emotion survey.csv")
            cyberbullying_path = os.path.join(self.data_dir, "label_data/vine_labeled_cyberbullying_data.csv")
            
            print(f"Loading emotion data from: {emotion_path}")
            self.emotion_data = pd.read_csv(emotion_path)
            
            print(f"Loading cyberbullying data from: {cyberbullying_path}")
            self.cyberbullying_data = pd.read_csv(cyberbullying_path)
            
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
        """Extract media session nodes"""
        for post_id, post_info in self.video_data.items():
            # Get emotion and theme annotation from aggregate video emotion survey.csv
            video_url = post_info.get('permalinkUrl', '')
            emotion_row = self.emotion_data[self.emotion_data['videolink'] == video_url + '/embed/simple']
            emotion_data = emotion_row.iloc[0] if not emotion_row.empty else None
            
            # Get cyberbullying annotation from vine_labeled_cyberbullying_data.csv
            cyberbullying_row = self.cyberbullying_data[self.cyberbullying_data['videolink'] == video_url + '/embed/simple']
            cyberbullying_data = cyberbullying_row.iloc[0] if not cyberbullying_row.empty else None
            
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
                    # 从 aggregate video emotion survey.csv 读取情感和主题标注
                    "emotion_annotation": {
                        "emotion": emotion_data["question2"] if emotion_data is not None and "question2" in emotion_data else None,
                        "theme": emotion_data["question3"] if emotion_data is not None and "question3" in emotion_data else None,
                        "emotion_confidence": emotion_data["question2:confidence"] if emotion_data is not None and "question2:confidence" in emotion_data else None,
                        "theme_confidence": emotion_data["question3:confidence"] if emotion_data is not None and "question3:confidence" in emotion_data else None
                    } if emotion_data is not None else None,
                    # 从 vine_labeled_cyberbullying_data.csv 读取攻击性和霸凌标注
                    "cyberbullying_annotation": {
                        "aggression": cyberbullying_data["question1"] if cyberbullying_data is not None and "question1" in cyberbullying_data else None,
                        "bullying": cyberbullying_data["question2"] if cyberbullying_data is not None and "question2" in cyberbullying_data else None,
                        "aggression_confidence": cyberbullying_data["question1:confidence"] if cyberbullying_data is not None and "question1:confidence" in cyberbullying_data else None,
                        "bullying_confidence": cyberbullying_data["question2:confidence"] if cyberbullying_data is not None and "question2:confidence" in cyberbullying_data else None
                    } if cyberbullying_data is not None else None
                }
            }
            self.nodes["media_sessions"].append(media_session)

    def extract_comments(self):
        """Extract comment nodes and relationships"""
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
                "id": full_id,  # 使用完整ID作为评论节点的ID
                "type": "comment",
                "properties": {
                    "text": comment_text,
                    "postId": post_id,
                    "commentId": comment_id
                }
            }
            self.nodes["comments"].append(comment_node)
            
            # Add comment relationship
            comment_rel = {
                "source": full_id,  # 使用完整ID作为来源
                "target": post_id,  # 使用postID作为目标
                "type": "comments_on",
                "properties": {}
            }
            self.relationships["comments"].append(comment_rel)

    def extract_relationships(self):
        """Extract all relationships"""
        for post_id, post_info in self.video_data.items():
            user_id = post_info.get("userId", "")
            
            # Add post relationship
            post_rel = {
                "source": user_id,
                "target": post_id,
                "type": "posts",
                "properties": {
                    "timestamp": post_info.get("created", ""),
                    "description": post_info.get("description", ""),
                }
            }
            self.relationships["posts"].append(post_rel)
            
            # Add interaction relationships (likes, loops)
            if user_id:
                interaction_rel = {
                    "source": user_id,
                    "target": post_id,
                    "type": "interacts_with",
                    "properties": {
                        "likeCount": post_info.get("likeCount", 0),
                        "loopCount": post_info.get("loopCount", 0),
                        "repostCount": post_info.get("repostCount", 0),
                        "commentCount": post_info.get("commentCount", 0),
                        "timestamp": post_info.get("created", "")
                    }
                }
                self.relationships["interactions"].append(interaction_rel)

    def save_to_json(self, output_file: str = "data/processed/combined_features.json"):
        """Save all extracted features to a JSON file"""
        combined_data = {
            "nodes": {
                "comments": self.nodes["comments"],
                "annotations": self.nodes["annotations"],
                "media_sessions": self.nodes["media_sessions"],
                "users": self.nodes["users"]
            },
            "relationships": {
                "posts": self.relationships["posts"],
                "comments": self.relationships["comments"],
                "interactions": self.relationships["interactions"]
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
        self.extract_relationships()
        self.save_to_json()

if __name__ == "__main__":
    extractor = FeatureExtractor()
    extractor.extract_all()
