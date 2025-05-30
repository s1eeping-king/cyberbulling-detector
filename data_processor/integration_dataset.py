import json
import pandas as pd
import os
from typing import Dict, List, Any

class IntegratedFeatureExtractor:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.video_data: Dict[str, Any] = {}
        self.comment_data: Dict[str, Any] = {}
        self.user_data: Dict[str, Any] = {}
        self.emotion_data: pd.DataFrame = pd.DataFrame()
        self.cyberbullying_data: pd.DataFrame = pd.DataFrame()
        self.url_to_postid: Dict[str, str] = {}

    def load_json_file(self, file_path: str) -> Dict:
        """Load a JSON file that may contain multiple JSON objects"""
        print(f"Loading JSON file from: {file_path}")
        result = {}
        try:
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
        except FileNotFoundError:
            print(f"Error: File not found at {file_path}")
            raise
        except Exception as e:
            print(f"An unexpected error occurred while loading {file_path}: {e}")
            raise
        return result

    def load_data(self):
        """Load necessary data sources for integration"""
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
                next(f) # Skip header
                for line in f:
                    line = line.strip()
                    if not line: continue
                    try:
                        postid, url = line.split(',')
                        self.url_to_postid[url] = postid
                        if not url.endswith('/embed/simple'):
                            self.url_to_postid[url + '/embed/simple'] = postid
                    except ValueError:
                         print(f"Skipping malformed line in url mapping: {line}")
                         continue

            # Load label data
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

    def extract_integrated_media_sessions(self) -> List[Dict[str, Any]]:
        """Extract media session features including base properties and all labels."""
        integrated_media_sessions = []
        video_labels = {}

        # Process emotion data
        for _, row in self.emotion_data.iterrows():
            video_url = row.get('videolink')
            if pd.isna(video_url):
                continue
            post_id = self.url_to_postid.get(video_url)
            if not post_id:
                # print(f"Warning: No postID found for emotion URL: {video_url}")
                continue
            video_labels[post_id] = {
                "emotion": row.get("question2"),
                "theme": row.get("question3"),
                "emotion_confidence": row.get("question2:confidence"),
                "theme_confidence": row.get("question3:confidence")
            }

        # Process cyberbullying data
        for _, row in self.cyberbullying_data.iterrows():
            video_url = row.get('videolink')
            if pd.isna(video_url):
                continue
            post_id = self.url_to_postid.get(video_url)
            if not post_id:
                # print(f"Warning: No postID found for cyberbullying URL: {video_url}")
                continue

            label_info = {
                "aggression": row.get("question1"),
                "bullying": row.get("question2"),
                "aggression_confidence": row.get("question1:confidence"),
                "bullying_confidence": row.get("question2:confidence")
            }
            if post_id in video_labels:
                video_labels[post_id].update(label_info)
            else:
                video_labels[post_id] = label_info

        # Create integrated media session data
        for post_id, post_info in self.video_data.items():
            labels = video_labels.get(post_id, {})
            media_session_features = {
                "postId": post_id,
                "permalinkUrl": post_info.get("permalinkUrl", ""),
                "description": post_info.get("description", ""),
                "likeCount": post_info.get("likeCount", 0),
                "commentCount": post_info.get("commentCount", 0),
                "loopCount": post_info.get("loopCount", 0),
                "repostCount": post_info.get("repostCount", 0),
                "created": post_info.get("created", ""),
                "userId": post_info.get("userId", ""),
                "username": post_info.get("username", ""),
                "emotion": labels.get("emotion"),
                "theme": labels.get("theme"),
                "emotion_confidence": labels.get("emotion_confidence"),
                "theme_confidence": labels.get("theme_confidence"),
                "aggression": labels.get("aggression"),
                "bullying": labels.get("bullying"),
                "aggression_confidence": labels.get("aggression_confidence"),
                "bullying_confidence": labels.get("bullying_confidence")
            }
            integrated_media_sessions.append(media_session_features)

        return integrated_media_sessions

    def extract_integrated_users(self) -> List[Dict[str, Any]]:
        """Extract user features including base properties."""
        integrated_users = []
        for user_id, user_info in self.user_data.items():
            user_features = {
                "userId": user_id,
                "username": user_info.get("username", ""),
                "description": user_info.get("description", ""),
                "followerCount": user_info.get("followerCount", 0),
                "followingCount": user_info.get("followingCount", 0),
                "likeCount": user_info.get("likeCount", 0),
                "postCount": user_info.get("postCount", 0)
            }
            integrated_users.append(user_features)
        return integrated_users

    def extract_integrated_comments(self) -> List[Dict[str, Any]]:
        """Extract comment features including base properties."""
        integrated_comments = []
        for full_id, comment_info in self.comment_data.items():
            try:
                post_id, comment_id = full_id.split('_')
            except ValueError:
                print(f"Warning: Invalid comment ID format: {full_id}")
                continue

            comment_text = comment_info.get("commentText", "")
            if not comment_text or not isinstance(comment_text, str):
                continue

            comment_features = {
                "commentId": comment_id,
                "postId": post_id,
                "text": comment_text,
                "userId": comment_info.get("userId", ""),
                "created": comment_info.get("created", "") # Assuming 'created' field exists
            }

            # 处理mention属性
            if comment_info.get("type") == "mention":
                link = comment_info.get("link", "")
                if link and "vine://user-id/" in link:
                    # 从链接中提取用户ID（最后的数字串部分）
                    mentionId = link.split("/")[-1]
                    comment_features["mentionId"] = mentionId

            integrated_comments.append(comment_features)
        return integrated_comments

    def save_to_json(self, data: List[Dict[str, Any]], output_file: str):
        """Save the extracted data to a JSON file."""
        output_dir = os.path.dirname(output_file)
        os.makedirs(output_dir, exist_ok=True)
        print(f"Saving data to: {output_file}")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Successfully saved {len(data)} items.")

if __name__ == "__main__":
    # Define the base data directory relative to the script location or use an absolute path
    # Assuming the script is run from the root of the project 'multimodal_v6'
    base_data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    output_dir = os.path.join(base_data_dir, 'processed', 'integration')

    extractor = IntegratedFeatureExtractor(data_dir=base_data_dir)
    extractor.load_data()

    # Extract and save media sessions
    media_sessions_data = extractor.extract_integrated_media_sessions()
    media_sessions_output_file = os.path.join(output_dir, "media_sessions.json")
    extractor.save_to_json(media_sessions_data, media_sessions_output_file)

    # Extract and save users
    print("\nExtracting users...")
    users_data = extractor.extract_integrated_users()
    users_output_file = os.path.join(output_dir, "users.json")
    extractor.save_to_json(users_data, users_output_file)

    # Extract and save comments
    print("\nExtracting comments...")
    comments_data = extractor.extract_integrated_comments()
    comments_output_file = os.path.join(output_dir, "comments.json")
    extractor.save_to_json(comments_data, comments_output_file)

    print("\nProcessing complete.")