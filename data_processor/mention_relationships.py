import json
import os
import spacy
from spacy.tokens import Doc
from typing import Dict, List, Any, Tuple, Set
import re

class MentionProcessor:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.mentions = []
        print("Loading spaCy model...")
        self.nlp = spacy.load("en_core_web_sm")
        
        # 常见非人名词汇列表
        self.non_name_words = {
            'youtube', 'instagram', 'facebook', 'twitter',
            'shit', 'fuck', 'damn', 'hell', 'lmao', 'lol',
            'the', 'and', 'but', 'or', 'if', 'then',
            'video', 'post', 'share', 'like', 'comment',
            'good', 'bad', 'nice', 'cool', 'great', 'awesome',
            'follow', 'following', 'followed', 'followers',
            'subscribe', 'watch', 'watching', 'watched',
            'click', 'link', 'links', 'url', 'website'
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
        """Load comment and user data"""
        try:
            # Load comment data
            comment_path = os.path.join(self.data_dir, "sampled_post-comments_vine.json")
            self.comment_data = self.load_json_file(comment_path)
        
            # Load user data
            user_path = os.path.join(self.data_dir, "vine_users_data.json")
            self.user_data = self.load_json_file(user_path)
            
        except FileNotFoundError as e:
            print(f"Error: Could not find file: {e}")
            raise
        except Exception as e:
            print(f"Unexpected error while loading data: {e}")
            raise

    def is_valid_username(self, username: str) -> bool:
        """检查用户名是否有效"""
        username = username.lower().strip()
        
        # 过滤掉纯特殊字符的用户名
        if all(not c.isalnum() for c in username):
            return False
            
        # 过滤掉常见非人名词汇
        if username in self.non_name_words:
            return False
            
        # 过滤掉过短的用户名
        if len(username) < 2:
            return False
            
        return True

    def extract_potential_mentions(self, text: str, doc: Doc) -> Set[Tuple[str, Tuple[int, int], str]]:
        """
        使用多种方法提取可能的提及
        返回: 集合的元素为 (提及文本, (开始位置, 结束位置), 置信度)
        """
        # 使用字典来跟踪每个位置的提及，保存最高置信度
        mention_dict = {}
        
        def add_mention(text: str, span: Tuple[int, int], confidence: str):
            """Helper function to add mention with confidence ranking"""
            confidence_rank = {"high": 3, "medium": 2, "low": 1}
            key = (text.lower(), span)
            current_conf = mention_dict.get(key, ("", 0))[1]
            new_conf = confidence_rank[confidence]
            if new_conf > current_conf:
                mention_dict[key] = (text, new_conf, confidence)
        
        # 1. 使用NER识别人名
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                add_mention(ent.text, (ent.start_char, ent.end_char), "high")
        
        # 2. 查找@开头的提及
        for match in re.finditer(r'@\w+', text):
            add_mention(match.group()[1:], (match.start()+1, match.end()), "high")
        
        # 3. 识别专有名词序列
        for chunk in doc.noun_chunks:
            if chunk.root.pos_ == "PROPN" and self.validate_mention_context(doc, chunk.start_char, chunk.end_char):
                add_mention(chunk.text, (chunk.start_char, chunk.end_char), "medium")
        
        # 转换回集合格式，只包含最高置信度的提及
        return {(text, span, conf) for (_, span), (text, _, conf) in mention_dict.items()}

    def validate_mention_context(self, doc: Doc, start_char: int, end_char: int) -> bool:
        """验证提及的上下文是否适合作为人名"""
        # 找到对应的token
        start_token = None
        end_token = None
        for token in doc:
            if token.idx <= start_char < token.idx + len(token.text):
                start_token = token
            if token.idx <= end_char <= token.idx + len(token.text):
                end_token = token
                break
        
        if not (start_token and end_token):
            return False
            
        # 检查词性标注
        mention_span = doc[start_token.i:end_token.i+1]
        
        # 如果是专有名词，可能性更高
        if all(t.pos_ == "PROPN" for t in mention_span):
            return True
            
        # 检查依存关系
        # 如果作为主语或宾语，可能性更高
        if any(t.dep_ in {"nsubj", "dobj", "pobj"} for t in mention_span):
            return True
            
        # 如果前面有称谓词，可能性更高
        if start_token.i > 0:
            prev_token = doc[start_token.i - 1]
            if prev_token.text.lower() in {"mr", "mrs", "ms", "miss", "dr", "prof"}:
                return True
        
        return False

    def process_mentions(self):
        """Process mentions and create mention relationships"""
        # 创建用户名到用户ID的映射
        username_to_userid = {}
        for user_id, user_info in self.user_data.items():
            username = user_info.get("username", "").strip()
            if self.is_valid_username(username):
                username_to_userid[username.lower()] = {
                    'id': user_id,
                    'original': username
                }
        
        # 使用字典来跟踪每个评论中的唯一提及
        processed_mentions = {}
        
        for comment_id, comment_data in self.comment_data.items():
            try:
                post_id, comment_id = comment_id.split('_')
            except ValueError:
                continue
            
            # 只处理 type 为 mention 的评论
            if comment_data.get("type") == "mention":
                comment_text = comment_data.get("commentText", "")
                if not comment_text or not isinstance(comment_text, str):
                    continue
                
                # 使用spaCy处理文本
                doc = self.nlp(comment_text)
                
                # 提取可能的提及
                potential_mentions = self.extract_potential_mentions(comment_text, doc)
                
                # 验证提及是否匹配已知用户
                mention_key = (post_id, comment_id)
                if mention_key not in processed_mentions:
                    processed_mentions[mention_key] = set()
                
                for mention_text, span, confidence in potential_mentions:
                    mention_lower = mention_text.lower()
                    if mention_lower in username_to_userid and mention_lower not in processed_mentions[mention_key]:
                        user_info = username_to_userid[mention_lower]
                        mention = {
                            "comment_id": comment_id,
                            "post_id": post_id,
                            "target_user_id": user_info['id'],
                            "username": user_info['original'],
                            "comment_text": comment_text,
                            "match_position": list(span),
                            "confidence": confidence
                        }
                        self.mentions.append(mention)
                        processed_mentions[mention_key].add(mention_lower)

    def save_to_json(self, output_file: str = "data/processed/mention_relationships.json"):
        """Save mention relationships to a JSON file"""
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.mentions, f, indent=2, ensure_ascii=False)

    def process_all(self):
        """Run all processing steps"""
        self.load_data()
        self.process_mentions()
        self.save_to_json()

if __name__ == "__main__":
    processor = MentionProcessor()
    processor.process_all()
