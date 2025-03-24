import json
import os
from typing import Dict, List, Set

class FullFeatureCleaner:
    def __init__(self, input_file: str = "data/processed/combined_features.json",
                 output_file: str = "data/processed/full_features.json"):
        self.input_file = input_file
        self.output_file = output_file
        self.data = None
        self.comment_user_ids = set()  # 评论中出现的所有用户ID
        self.publisher_user_ids = set()  # 发布视频的用户ID
        self.existing_user_ids = set()  # 用户节点中已存在的用户ID
        self.valid_comment_ids = set()  # 有效的评论ID
        self.user_properties_template = None  # 用户节点属性模板

    def load_data(self):
        """加载原始数据"""
        print(f"Loading data from {self.input_file}")
        with open(self.input_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        # 获取用户属性模板
        if self.data['nodes']['users']:
            # 移除userId字段，因为这是冗余的
            properties = self.data['nodes']['users'][0]['properties'].copy()
            if 'userId' in properties:
                del properties['userId']
            self.user_properties_template = properties
            print("Extracted user properties template")

    def collect_user_ids(self):
        """收集评论中的用户ID和现有用户节点的ID"""
        # 1. 收集发布视频的用户ID
        self.publisher_user_ids = {
            rel['source'] 
            for rel in self.data['relationships']['publishes']
        }
        print(f"Found {len(self.publisher_user_ids)} users who published videos")

        # 2. 收集评论中的用户ID
        self.comment_user_ids = {
            comment['properties']['userId'] 
            for comment in self.data['nodes']['comments']
            if 'userId' in comment['properties']
        }
        print(f"Found {len(self.comment_user_ids)} unique user IDs in comments")

        # 3. 收集现有用户节点的ID
        self.existing_user_ids = {user['id'] for user in self.data['nodes']['users']}
        print(f"Found {len(self.existing_user_ids)} existing users in user nodes")

    def create_default_user_properties(self):
        """创建默认的用户属性"""
        default_properties = {}
        for key, value in self.user_properties_template.items():
            if isinstance(value, (int, float)):
                default_properties[key] = 0
            elif isinstance(value, bool):
                default_properties[key] = False
            elif isinstance(value, str):
                default_properties[key] = f"user_{key}"
            elif isinstance(value, list):
                default_properties[key] = []
            else:
                default_properties[key] = None
        return default_properties

    def process_user_nodes(self):
        """处理用户节点"""
        # 1. 标记现有的匹配用户节点
        filtered_users = []
        for user in self.data['nodes']['users']:
            # 如果用户在评论中出现或是发布者，则保留
            if user['id'] in self.comment_user_ids or user['id'] in self.publisher_user_ids:
                user['properties']['exist'] = True
                filtered_users.append(user)
        
        # 2. 创建缺失的用户节点
        all_required_user_ids = self.comment_user_ids.union(self.publisher_user_ids)
        missing_user_ids = all_required_user_ids - self.existing_user_ids
        default_properties = self.create_default_user_properties()
        
        for user_id in missing_user_ids:
            new_user = {
                'id': user_id,
                'type': 'User',  # 使用type而不是label
                'properties': default_properties.copy()
            }
            new_user['properties']['exist'] = False
            filtered_users.append(new_user)
        
        # 更新用户节点
        self.data['nodes']['users'] = filtered_users
        print(f"Updated user nodes: {len(filtered_users)} total ({len(missing_user_ids)} new)")
        print(f"  - Preserved {len(self.publisher_user_ids)} publisher users")
        print(f"  - Preserved {len(self.comment_user_ids)} commenter users")
        print(f"  - Added {len(missing_user_ids)} new users")

    def filter_relationships(self):
        """过滤和更新关系"""
        valid_user_ids = {user['id'] for user in self.data['nodes']['users']}
        
        # 1. 过滤creates关系
        filtered_creates = [rel for rel in self.data['relationships']['creates']
                          if rel['source'] in valid_user_ids]
        print(f"Filtered creates relationships: {len(filtered_creates)} (from {len(self.data['relationships']['creates'])})")
        
        # 2. 过滤mentions关系
        filtered_mentions = [rel for rel in self.data['relationships']['mentions']
                           if rel['target'] in valid_user_ids]
        print(f"Filtered mentions relationships: {len(filtered_mentions)} (from {len(self.data['relationships']['mentions'])})")
        
        # 3. 保留所有publishes关系（因为发布者已经被保留）
        filtered_publishes = self.data['relationships']['publishes']
        print(f"Preserved all publishes relationships: {len(filtered_publishes)}")
        
        # 更新关系
        self.data['relationships']['creates'] = filtered_creates
        self.data['relationships']['mentions'] = filtered_mentions
        self.data['relationships']['publishes'] = filtered_publishes

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
            if node_type == 'users':
                existing_users = sum(1 for user in nodes if user['properties'].get('exist', False))
                publisher_users = sum(1 for user in nodes if user['id'] in self.publisher_user_ids)
                print(f"  {node_type}: {len(nodes)} total")
                print(f"    - {existing_users} existing")
                print(f"    - {publisher_users} publishers")
                print(f"    - {len(nodes) - existing_users} new")
            else:
                print(f"  {node_type}: {len(nodes)}")
        print("\nRelationships:")
        for rel_type, rels in self.data['relationships'].items():
            print(f"  {rel_type}: {len(rels)}")

    def process(self):
        """执行完整的处理流程"""
        print("Starting data cleaning process...")
        self.load_data()
        self.collect_user_ids()
        self.process_user_nodes()
        self.filter_relationships()
        self.print_statistics()
        self.save_data()
        print("Data cleaning completed!")

def main():
    cleaner = FullFeatureCleaner()
    cleaner.process()

if __name__ == "__main__":
    main()
