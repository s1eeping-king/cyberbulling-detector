import json
import os
from typing import Dict, List, Set

class CoreFeatureCleaner:
    def __init__(self, input_file: str = "data/processed/combined_features.json",
                 output_file: str = "data/processed/core_features.json"):
        self.input_file = input_file
        self.output_file = output_file
        self.data = None
        self.valid_user_ids = set()  # 存储有效的用户ID
        self.valid_comment_ids = set()  # 存储有效的评论ID
        self.existing_user_ids = set()  # 存储用户节点中已存在的用户ID
        self.valid_media_ids = set()  # 存储有效的媒体会话ID
        
    def load_data(self):
        """加载原始数据"""
        print(f"Loading data from {self.input_file}")
        with open(self.input_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        # 加载现有的用户ID
        self.existing_user_ids = {user['id'] for user in self.data['nodes']['users']}
        print(f"Found {len(self.existing_user_ids)} existing users in user nodes")

    def identify_valid_users(self):
        """识别需要保留的用户ID"""
        # 1. 获取所有发布视频的用户ID（这些用户必须保留）
        publisher_user_ids = {rel['source'] for rel in self.data['relationships']['publishes']}
        print(f"Found {len(publisher_user_ids)} users who published videos")
        
        # 2. 获取评论中的用户ID（必须同时存在于用户节点中）
        comment_user_ids = set()
        for comment in self.data['nodes']['comments']:
            user_id = comment['properties'].get('userId')
            if user_id and user_id in self.existing_user_ids:
                comment_user_ids.add(user_id)
        
        # 3. 合并所有有效的用户ID（发布者 + 在用户节点中存在的评论用户）
        self.valid_user_ids = publisher_user_ids.union(comment_user_ids)
        print(f"Total valid users (publishers + valid commenters): {len(self.valid_user_ids)}")
        
        # 4. 记录有效的评论ID（评论的用户ID必须在有效用户列表中）
        self.valid_comment_ids = {
            comment['id'] 
            for comment in self.data['nodes']['comments']
            if comment['properties'].get('userId') in self.valid_user_ids
        }
        print(f"Valid comments after filtering: {len(self.valid_comment_ids)}")

    def filter_nodes(self):
        """过滤节点"""
        # 1. 过滤用户节点（只保留有效用户ID的节点）
        filtered_users = [user for user in self.data['nodes']['users'] 
                        if user['id'] in self.valid_user_ids]
        print(f"Filtered users: {len(filtered_users)} (from {len(self.data['nodes']['users'])})")
        
        # 2. 过滤评论节点（只保留有效评论ID的节点）
        filtered_comments = [comment for comment in self.data['nodes']['comments'] 
                           if comment['id'] in self.valid_comment_ids]
        print(f"Filtered comments: {len(filtered_comments)} (from {len(self.data['nodes']['comments'])})")
        
        # 更新节点
        self.data['nodes']['users'] = filtered_users
        self.data['nodes']['comments'] = filtered_comments

    def filter_relationships(self):
        """过滤关系"""
        # 1. 过滤creates关系（源节点必须是有效用户，目标节点必须是有效评论）
        filtered_creates = [rel for rel in self.data['relationships']['creates']
                          if rel['source'] in self.valid_user_ids and 
                             rel['target'] in self.valid_comment_ids]
        print(f"Filtered creates relationships: {len(filtered_creates)} (from {len(self.data['relationships']['creates'])})")
        
        # 2. 过滤mentions关系（源节点必须是有效评论，目标节点必须是有效用户）
        filtered_mentions = [rel for rel in self.data['relationships']['mentions']
                           if rel['source'] in self.valid_comment_ids and 
                              rel['target'] in self.valid_user_ids]
        print(f"Filtered mentions relationships: {len(filtered_mentions)} (from {len(self.data['relationships']['mentions'])})")
        
        # 3. 过滤belongs_to关系（源节点必须是有效评论）
        filtered_belongs = [rel for rel in self.data['relationships']['belongs_to']
                          if rel['source'] in self.valid_comment_ids]
        print(f"Filtered belongs_to relationships: {len(filtered_belongs)} (from {len(self.data['relationships']['belongs_to'])})")
        
        # 更新关系
        self.data['relationships']['creates'] = filtered_creates
        self.data['relationships']['mentions'] = filtered_mentions
        self.data['relationships']['belongs_to'] = filtered_belongs

    def save_data(self):
        """保存处理后的数据"""
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)
        print(f"Saved cleaned data to {self.output_file}")

    def print_statistics(self):
        """打印数据统计信息"""
        print("\nFinal Statistics:")
        print("Nodes:")
        for node_type, nodes in self.data['nodes'].items():
            print(f"  {node_type}: {len(nodes)}")
        print("\nRelationships:")
        for rel_type, rels in self.data['relationships'].items():
            print(f"  {rel_type}: {len(rels)}")

    def filter_media_sessions(self):
        """过滤没有annotates关系的媒体会话"""
        # 获取所有有annotates关系的媒体会话ID
        media_with_annotates = {rel['source'] for rel in self.data['relationships']['annotates']}
        print(f"Found {len(media_with_annotates)} media sessions with annotates relationships")
        
        # 过滤媒体会话节点
        original_count = len(self.data['nodes']['media_sessions'])
        self.valid_media_ids = media_with_annotates
        filtered_media = [media for media in self.data['nodes']['media_sessions'] 
                         if media['id'] in self.valid_media_ids]
        
        # 更新媒体会话节点
        self.data['nodes']['media_sessions'] = filtered_media
        print(f"Filtered media sessions: {len(filtered_media)} (from {original_count})")
        
        # 过滤相关的关系
        # 1. 过滤publishes关系
        filtered_publishes = [rel for rel in self.data['relationships']['publishes']
                            if rel['target'] in self.valid_media_ids]
        print(f"Filtered publishes relationships: {len(filtered_publishes)} (from {len(self.data['relationships']['publishes'])})")
        
        # 2. 过滤belongs_to关系
        filtered_belongs = [rel for rel in self.data['relationships']['belongs_to']
                          if rel['target'] in self.valid_media_ids]
        print(f"Filtered belongs_to relationships: {len(filtered_belongs)} (from {len(self.data['relationships']['belongs_to'])})")
        
        # 3. 过滤annotates关系（只保留源节点在有效媒体会话中的关系）
        original_annotates_count = len(self.data['relationships']['annotates'])
        filtered_annotates = [rel for rel in self.data['relationships']['annotates']
                             if rel['source'] in self.valid_media_ids]
        self.data['relationships']['annotates'] = filtered_annotates
        print(f"Filtered annotates relationships: {len(filtered_annotates)} (from {original_annotates_count})")
        
        # 4. 获取有效的标签ID（通过过滤后的annotates关系的目标节点）
        valid_label_ids = {rel['target'] for rel in filtered_annotates}
        
        # 5. 过滤labels节点（只保留ID在有效标签ID集合中的标签）
        original_labels_count = len(self.data['nodes']['labels'])
        filtered_labels = [label for label in self.data['nodes']['labels'] 
                          if label['id'] in valid_label_ids]
        self.data['nodes']['labels'] = filtered_labels
        print(f"Filtered labels: {len(filtered_labels)} (from {original_labels_count})")
        
        # 更新关系
        self.data['relationships']['publishes'] = filtered_publishes
        self.data['relationships']['belongs_to'] = filtered_belongs

    def process(self):
        """执行完整的处理流程"""
        print("Starting data cleaning process...")
        self.load_data()
        self.filter_media_sessions()  # 先过滤媒体会话
        self.identify_valid_users()
        self.filter_nodes()
        self.filter_relationships()
        self.print_statistics()
        self.save_data()
        print("Data cleaning completed!")

def main():
    cleaner = CoreFeatureCleaner()
    cleaner.process()

if __name__ == "__main__":
    main()
