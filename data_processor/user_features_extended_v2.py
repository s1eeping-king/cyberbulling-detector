import os
import csv
import json
from neo4j import GraphDatabase
from typing import Dict, List, Any, Tuple
import logging
from tqdm import tqdm
import numpy as np

class UserFeaturesExtractor:
    """从Neo4j数据库中提取用户特征"""

    def __init__(self, uri: str, username: str, password: str):
        """初始化用户特征提取器

        Args:
            uri: Neo4j数据库URI
            username: 数据库用户名
            password: 数据库密码
        """
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
        self.logger = self.setup_logger()

        # 确保输出目录存在
        os.makedirs("data/processed/features_extended", exist_ok=True)

    @staticmethod
    def setup_logger():
        """设置日志记录器"""
        logger = logging.getLogger('UserFeaturesExtractor')
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def close(self):
        """关闭Neo4j驱动连接"""
        self.driver.close()

    def get_users_with_mention_relationships(self) -> List[str]:
        """获取有OFFENSIVE_COMMENT或NON_OFFENSIVE_COMMENT关系的用户ID列表（用户之间的mention关系）"""
        query = """
        MATCH (u:User)-[r:OFFENSIVE_COMMENT|NON_OFFENSIVE_COMMENT]-()
        RETURN DISTINCT u.id AS userId
        UNION
        MATCH (u:User)<-[r:OFFENSIVE_COMMENT|NON_OFFENSIVE_COMMENT]-()
        RETURN DISTINCT u.id AS userId
        """

        with self.driver.session() as session:
            result = session.run(query)
            user_ids = [record["userId"] for record in result]

        self.logger.info(f"找到 {len(user_ids)} 个有mention关系的用户")
        return user_ids



    def extract_user_features(self, user_ids: List[str]) -> List[Dict[str, Any]]:
        """提取用户特征

        Args:
            user_ids: 用户ID列表

        Returns:
            List[Dict[str, Any]]: 用户特征列表
        """
        all_user_features = []

        for user_id in tqdm(user_ids, desc="提取用户特征"):
            # 获取用户基本特征
            user_features = self.get_user_basic_features(user_id)

            if not user_features:
                self.logger.warning(f"用户 {user_id} 未找到基本特征，跳过")
                continue

            # 获取mention相关特征
            mention_features = self.get_user_mention_features(user_id)
            user_features.update(mention_features)

            # 计算比例特征
            ratio_features = self.calculate_ratio_features(user_features)
            user_features.update(ratio_features)

            all_user_features.append(user_features)

        return all_user_features

    def get_user_basic_features(self, user_id: str) -> Dict[str, Any]:
        """获取用户基本特征

        Args:
            user_id: 用户ID

        Returns:
            Dict[str, Any]: 用户基本特征
        """
        query = """
        MATCH (u:User {id: $userId})
        RETURN u.id AS userId,
               u.description_offensive AS description_offensive,
               u.followerCount AS followerCount,
               u.followingCount AS followingCount,
               u.likeCount AS likeCount,
               u.postCount AS postCount
        """

        with self.driver.session() as session:
            result = session.run(query, userId=user_id).single()

            if not result:
                return {}

            # 处理description_offensive字段
            description_offensive = result["description_offensive"]
            if description_offensive == "offensive":
                description_offensive_value = 1
            else:
                description_offensive_value = 0

            return {
                "userId": result["userId"],
                "description_offensive": description_offensive_value,
                "follower_count": int(result["followerCount"] or 0),
                "following_count": int(result["followingCount"] or 0),
                "like_count": int(result["likeCount"] or 0),
                "post_count": int(result["postCount"] or 0)
            }

    def get_user_mention_features(self, user_id: str) -> Dict[str, Any]:
        """获取用户mention相关特征

        Args:
            user_id: 用户ID

        Returns:
            Dict[str, Any]: 用户mention相关特征
        """
        # 1. mention次数 (用户发出的总mention数量)
        # 2. 被mention次数 (用户收到的总mention数量)
        # 3. 被OFFENSIVE次数 (用户收到的被判定为OFFENSIVE的mention数量)
        # 4. 被NON_OFFENSIVE次数 (用户收到的被判定为NON_OFFENSIVE的mention数量)
        # 5. OFFENSIVE mention次数 (用户发出的被判定为OFFENSIVE的mention数量)
        # 6. NON_OFFENSIVE mention次数 (用户发出的被判定为NON_OFFENSIVE的mention数量)
        query1 = """
        MATCH (u:User {id: $userId})-[r:OFFENSIVE_COMMENT]->()
        WITH count(r) AS offensive_count
        MATCH (u:User {id: $userId})-[r:NON_OFFENSIVE_COMMENT]->()
        WITH offensive_count, count(r) AS non_offensive_count
        RETURN offensive_count + non_offensive_count AS outgoing_mentions,
               offensive_count AS outgoing_offensive,
               non_offensive_count AS outgoing_non_offensive
        """

        query2 = """
        MATCH (u:User {id: $userId})<-[r:OFFENSIVE_COMMENT]-()
        WITH count(r) AS offensive_count
        MATCH (u:User {id: $userId})<-[r:NON_OFFENSIVE_COMMENT]-()
        WITH offensive_count, count(r) AS non_offensive_count
        RETURN offensive_count + non_offensive_count AS incoming_mentions,
               offensive_count AS incoming_offensive,
               non_offensive_count AS incoming_non_offensive
        """

        # 8. 被mention的用户数量 (该用户被多少个不同的用户mention过)
        # 9. mention的用户数量 (该用户mention过多少个不同的用户)
        query3 = """
        MATCH (u:User {id: $userId})-[r:OFFENSIVE_COMMENT|NON_OFFENSIVE_COMMENT]->(target:User)
        RETURN count(DISTINCT target) AS unique_targets
        """

        query4 = """
        MATCH (u:User {id: $userId})<-[r:OFFENSIVE_COMMENT|NON_OFFENSIVE_COMMENT]-(source:User)
        RETURN count(DISTINCT source) AS unique_sources
        """

        # 10. 对用户mention的最大次数 (该用户对单一目标用户mention的最大次数，不区分offensive/non-offensive)
        # 11. 对一位用户的最大OFFENSIVE次数 (该用户对单一目标用户发出OFFENSIVE mention的最大次数)
        # 12. 对一位用户的最大NON_OFFENSIVE次数 (该用户对单一目标用户发出NON_OFFENSIVE mention的最大次数)
        query5 = """
        MATCH (u:User {id: $userId})-[r:OFFENSIVE_COMMENT]->(target:User)
        WITH target, count(r) AS offensive_count
        MATCH (u:User {id: $userId})-[r:NON_OFFENSIVE_COMMENT]->(target:User)
        WITH target, offensive_count, count(r) AS non_offensive_count
        WITH target, offensive_count, non_offensive_count, offensive_count + non_offensive_count AS total_count
        RETURN max(total_count) AS max_mentions_to_single_user,
               max(offensive_count) AS max_offensive_to_single_user,
               max(non_offensive_count) AS max_non_offensive_to_single_user
        """

        # 19. 被攻击集中度（收到）(从单一用户收到的最大OFFENSIVE mention次数 / 总被OFFENSIVE mention次数)
        # 20. 被NON_OFFENSIVE mention集中度（收到）(从单一用户收到的最大NON_OFFENSIVE mention次数 / 总被NON_OFFENSIVE mention次数)
        query6 = """
        MATCH (u:User {id: $userId})<-[r:OFFENSIVE_COMMENT]-(source:User)
        WITH source, count(r) AS offensive_count
        MATCH (u:User {id: $userId})<-[r:NON_OFFENSIVE_COMMENT]-(source:User)
        WITH source, offensive_count, count(r) AS non_offensive_count
        RETURN max(offensive_count) AS max_offensive_from_single_user,
               max(non_offensive_count) AS max_non_offensive_from_single_user
        """

        # 21. 双向攻击用户数 (与该用户互发过（至少一次）OFFENSIVE mention的独特用户数量)
        # 22. 双向NON_OFFENSIVE互动用户数 (与该用户互发过（至少一次）NON_OFFENSIVE mention的独特用户数量)
        query7 = """
        MATCH (u:User {id: $userId})-[r1:OFFENSIVE_COMMENT]->(other:User)-[r2:OFFENSIVE_COMMENT]->(u)
        RETURN count(DISTINCT other) AS bidirectional_offensive_users
        """

        query8 = """
        MATCH (u:User {id: $userId})-[r1:NON_OFFENSIVE_COMMENT]->(other:User)-[r2:NON_OFFENSIVE_COMMENT]->(u)
        RETURN count(DISTINCT other) AS bidirectional_non_offensive_users
        """

        # 23. 未反击比例 (衡量用户收到OFFENSIVE mention后未进行OFFENSIVE反击的程度或比例)
        query9 = """
        MATCH (source:User)-[r:OFFENSIVE_COMMENT]->(u:User {id: $userId})
        WHERE NOT EXISTS((u)-[:OFFENSIVE_COMMENT]->(source))
        RETURN count(DISTINCT source) AS unretaliated_users
        """

        with self.driver.session() as session:
            result1 = session.run(query1, userId=user_id).single() or {"outgoing_mentions": 0, "outgoing_offensive": 0, "outgoing_non_offensive": 0}
            result2 = session.run(query2, userId=user_id).single() or {"incoming_mentions": 0, "incoming_offensive": 0, "incoming_non_offensive": 0}
            result3 = session.run(query3, userId=user_id).single() or {"unique_targets": 0}
            result4 = session.run(query4, userId=user_id).single() or {"unique_sources": 0}
            result5 = session.run(query5, userId=user_id).single() or {"max_mentions_to_single_user": 0, "max_offensive_to_single_user": 0, "max_non_offensive_to_single_user": 0}
            result6 = session.run(query6, userId=user_id).single() or {"max_offensive_from_single_user": 0, "max_non_offensive_from_single_user": 0}
            result7 = session.run(query7, userId=user_id).single() or {"bidirectional_offensive_users": 0}
            result8 = session.run(query8, userId=user_id).single() or {"bidirectional_non_offensive_users": 0}
            result9 = session.run(query9, userId=user_id).single() or {"unretaliated_users": 0}

            # 计算未反击比例
            incoming_offensive_users = session.run("""
                MATCH (source:User)-[r:OFFENSIVE_COMMENT]->(u:User {id: $userId})
                RETURN count(DISTINCT source) AS count
            """, userId=user_id).single()["count"]

            unretaliated_ratio = 0
            if incoming_offensive_users > 0:
                unretaliated_ratio = result9["unretaliated_users"] / incoming_offensive_users

            return {
                "mention_count": result1["outgoing_mentions"],
                "received_mention_count": result2["incoming_mentions"],
                "received_offensive_count": result2["incoming_offensive"],
                "received_non_offensive_count": result2["incoming_non_offensive"],
                "offensive_mention_count": result1["outgoing_offensive"],
                "non_offensive_mention_count": result1["outgoing_non_offensive"],
                "unique_mentioners_count": result4["unique_sources"],
                "unique_targets_count": result3["unique_targets"],
                "max_mentions_to_single_user": result5["max_mentions_to_single_user"],
                "max_offensive_to_single_user": result5["max_offensive_to_single_user"],
                "max_non_offensive_to_single_user": result5["max_non_offensive_to_single_user"],
                "max_offensive_from_single_user": result6["max_offensive_from_single_user"],
                "max_non_offensive_from_single_user": result6["max_non_offensive_from_single_user"],
                "bidirectional_offensive_users": result7["bidirectional_offensive_users"],
                "bidirectional_non_offensive_users": result8["bidirectional_non_offensive_users"],
                "unretaliated_ratio": unretaliated_ratio
            }



    def calculate_ratio_features(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """计算比例特征

        Args:
            features: 用户特征字典

        Returns:
            Dict[str, Any]: 比例特征字典
        """
        ratio_features = {}

        # 确保所有需要的特征都是数值类型
        for key in features:
            if features[key] is None:
                features[key] = 0

        # 13. 发出mention的OFFENSIVE比例
        mentions_count = int(features.get("mention_count", 0))
        if mentions_count > 0:
            ratio_features["offensive_mention_ratio"] = int(features.get("offensive_mention_count", 0)) / mentions_count
        else:
            ratio_features["offensive_mention_ratio"] = 0

        # 14. 发出mention的NON_OFFENSIVE比例
        if mentions_count > 0:
            ratio_features["non_offensive_mention_ratio"] = int(features.get("non_offensive_mention_count", 0)) / mentions_count
        else:
            ratio_features["non_offensive_mention_ratio"] = 0

        # 15. 收到mention的OFFENSIVE比例
        incoming_mentions = int(features.get("received_mention_count", 0))
        if incoming_mentions > 0:
            ratio_features["received_offensive_ratio"] = int(features.get("received_offensive_count", 0)) / incoming_mentions
        else:
            ratio_features["received_offensive_ratio"] = 0

        # 16. 收到mention的NON_OFFENSIVE比例
        if incoming_mentions > 0:
            ratio_features["received_non_offensive_ratio"] = int(features.get("received_non_offensive_count", 0)) / incoming_mentions
        else:
            ratio_features["received_non_offensive_ratio"] = 0

        # 17. 攻击集中度（发出）
        offensive_mentions = int(features.get("offensive_mention_count", 0))
        if offensive_mentions > 0:
            ratio_features["offensive_concentration_out"] = int(features.get("max_offensive_to_single_user", 0)) / offensive_mentions
        else:
            ratio_features["offensive_concentration_out"] = 0

        # 18. NON_OFFENSIVE mention集中度（发出）
        non_offensive_mentions = int(features.get("non_offensive_mention_count", 0))
        if non_offensive_mentions > 0:
            ratio_features["non_offensive_concentration_out"] = int(features.get("max_non_offensive_to_single_user", 0)) / non_offensive_mentions
        else:
            ratio_features["non_offensive_concentration_out"] = 0

        # 19. 被攻击集中度（收到）
        incoming_offensive = int(features.get("received_offensive_count", 0))
        if incoming_offensive > 0:
            ratio_features["offensive_concentration_in"] = int(features.get("max_offensive_from_single_user", 0)) / incoming_offensive
        else:
            ratio_features["offensive_concentration_in"] = 0

        # 20. 被NON_OFFENSIVE mention集中度（收到）
        incoming_non_offensive = int(features.get("received_non_offensive_count", 0))
        if incoming_non_offensive > 0:
            ratio_features["non_offensive_concentration_in"] = int(features.get("max_non_offensive_from_single_user", 0)) / incoming_non_offensive
        else:
            ratio_features["non_offensive_concentration_in"] = 0

        return ratio_features

    def save_features_to_csv(self, features: List[Dict[str, Any]], output_path: str):
        """将特征保存到CSV文件

        Args:
            features: 特征列表
            output_path: 输出文件路径
        """
        if not features:
            self.logger.warning("没有特征数据可保存")
            return

        # 获取所有特征列
        all_columns = set()
        for feature in features:
            all_columns.update(feature.keys())

        # 确保userId是第一列
        columns = ["userId"]
        for col in sorted(all_columns):
            if col != "userId":
                columns.append(col)

        self.logger.info(f"保存 {len(features)} 条用户特征记录到 {output_path}")

        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            writer.writerows(features)

        self.logger.info(f"特征已保存到 {output_path}")

def main():
    """主函数"""
    # Neo4j连接信息
    uri = "bolt://localhost:7687"
    username = "neo4j"
    password = "12345678"  # 请替换为实际密码

    # 输出文件路径
    output_path = "data/processed/features_extended/user_features_extended.csv"

    # 初始化特征提取器
    extractor = UserFeaturesExtractor(uri, username, password)

    try:
        # 获取有mention关系的用户
        user_ids = extractor.get_users_with_mention_relationships()

        # 提取用户特征
        user_features = extractor.extract_user_features(user_ids)

        # 保存特征到CSV文件
        extractor.save_features_to_csv(user_features, output_path)

    finally:
        # 关闭连接
        extractor.close()

if __name__ == "__main__":
    main()