import os
import csv
import json
from neo4j import GraphDatabase
from typing import Dict, List, Any, Tuple, Set
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
        """获取有MENTIONS关系的用户ID列表（用户之间的mention关系）"""
        query = """
        MATCH (u:User)-[:CREATES]->(c:Comment)-[:MENTIONS]->(:User)
        RETURN DISTINCT u.id AS userId
        UNION
        MATCH (c:Comment)-[:MENTIONS]->(u:User)
        RETURN DISTINCT u.id AS userId
        """

        with self.driver.session() as session:
            result = session.run(query)
            user_ids = [record["userId"] for record in result]

        self.logger.info(f"找到 {len(user_ids)} 个有mention关系的用户")
        return user_ids

    def get_all_users(self) -> List[str]:
        """获取所有用户ID列表"""
        query = """
        MATCH (u:User)
        RETURN DISTINCT u.id AS userId
        """

        with self.driver.session() as session:
            result = session.run(query)
            user_ids = [record["userId"] for record in result]

        self.logger.info(f"找到 {len(user_ids)} 个用户")
        return user_ids

    def get_media_sessions_by_time(self) -> List[Tuple[str, str]]:
        """按创建时间获取所有媒体会话ID

        Returns:
            List[Tuple[str, str]]: 媒体会话ID和创建时间的元组列表，按时间排序
        """
        query = """
        MATCH (m:MediaSession)
        WHERE m.created IS NOT NULL
        RETURN m.id AS mediaId, m.created AS created
        ORDER BY created
        """

        with self.driver.session() as session:
            result = session.run(query)
            media_sessions = [(record["mediaId"], record["created"]) for record in result]

        self.logger.info(f"找到 {len(media_sessions)} 个有创建时间的媒体会话")
        return media_sessions

    def get_users_with_comments(self) -> List[str]:
        """获取有评论的用户ID列表"""
        query = """
        MATCH (u:User)-[:CREATES]->(c:Comment)
        RETURN DISTINCT u.id AS userId
        """

        with self.driver.session() as session:
            result = session.run(query)
            user_ids = [record["userId"] for record in result]

        self.logger.info(f"找到 {len(user_ids)} 个有评论的用户")
        return user_ids

    def get_users_in_media_sessions(self, media_session_ids: List[str]) -> List[str]:
        """获取特定媒体会话中的用户ID列表

        Args:
            media_session_ids: 媒体会话ID列表

        Returns:
            List[str]: 用户ID列表
        """
        query = """
        MATCH (u:User)-[:CREATES]->(c:Comment)-[:BELONGS_TO]->(m:MediaSession)
        WHERE m.id IN $mediaIds
        RETURN DISTINCT u.id AS userId
        """

        with self.driver.session() as session:
            result = session.run(query, mediaIds=media_session_ids)
            user_ids = [record["userId"] for record in result]

        self.logger.info(f"在 {len(media_session_ids)} 个媒体会话中找到 {len(user_ids)} 个用户")
        return user_ids

    def get_comments_in_media_sessions(self, media_session_ids: List[str]) -> List[str]:
        """获取特定媒体会话中的评论ID列表

        Args:
            media_session_ids: 媒体会话ID列表

        Returns:
            List[str]: 评论ID列表
        """
        query = """
        MATCH (c:Comment)-[:BELONGS_TO]->(m:MediaSession)
        WHERE m.id IN $mediaIds
        RETURN DISTINCT c.id AS commentId
        """

        with self.driver.session() as session:
            result = session.run(query, mediaIds=media_session_ids)
            comment_ids = [record["commentId"] for record in result]

        self.logger.info(f"在 {len(media_session_ids)} 个媒体会话中找到 {len(comment_ids)} 条评论")
        return comment_ids



    def extract_user_features(self, user_ids: List[str], statistic_media_sessions: List[str] = None) -> List[Dict[str, Any]]:
        """提取用户特征

        Args:
            user_ids: 需要提取特征的用户ID列表
            statistic_media_sessions: 用于计算统计特征的媒体会话ID列表，如果为None，则不限制范围

        Returns:
            List[Dict[str, Any]]: 用户特征列表
        """
        all_user_features = []

        # 如果提供了statistic_media_sessions，则获取对应的评论集
        statistic_comment_ids = None
        if statistic_media_sessions is not None:
            statistic_comment_ids = set(self.get_comments_in_media_sessions(statistic_media_sessions))
            self.logger.info(f"使用 {len(statistic_media_sessions)} 个媒体会话中的 {len(statistic_comment_ids)} 条评论计算统计特征")

        # 获取有mention关系的用户ID集合
        users_with_mention = set(self.get_users_with_mention_relationships())
        self.logger.info(f"共有 {len(users_with_mention)} 个用户参与了mention关系")

        for user_id in tqdm(user_ids, desc="提取用户特征"):
            # 获取用户基本特征
            user_features = self.get_user_basic_features(user_id)

            if not user_features:
                self.logger.warning(f"用户 {user_id} 未找到基本特征，跳过")
                continue

            # 添加用户是否参与mention关系的标识
            user_features["has_mention_relationship"] = 1 if user_id in users_with_mention else 0

            # 获取mention相关特征（基于评论集）
            mention_features = self.get_user_mention_features(user_id, statistic_comment_ids)
            user_features.update(mention_features)

            # 获取评论相关特征（基于评论集）
            comment_features = self.get_user_comment_features(user_id, statistic_comment_ids)
            user_features.update(comment_features)

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

    def get_user_mention_features(self, user_id: str, statistic_comment_ids: Set[str] = None) -> Dict[str, Any]:
        """获取用户mention相关特征

        Args:
            user_id: 用户ID
            statistic_comment_ids: 用于计算统计特征的评论ID集合，如果为None，则不限制评论范围

        Returns:
            Dict[str, Any]: 用户mention相关特征
        """
        # 1. mention次数 (用户发出的总mention数量)
        # 2. 被mention次数 (用户收到的总mention数量)
        # 3. 被OFFENSIVE次数 (用户收到的被判定为OFFENSIVE的mention数量)
        # 4. 被NON_OFFENSIVE次数 (用户收到的被判定为NON_OFFENSIVE的mention数量)
        # 5. OFFENSIVE mention次数 (用户发出的被判定为OFFENSIVE的mention数量)
        # 6. NON_OFFENSIVE mention次数 (用户发出的被判定为NON_OFFENSIVE的mention数量)
        # 如果提供了statistic_comment_ids，则限制查询范围到指定的评论集
        if statistic_comment_ids is not None:
            # 1. mention次数 (用户发出的总mention数量) - 基于评论集
            query1 = """
            MATCH (u:User {id: $userId})-[:CREATES]->(c:Comment)-[:MENTIONS]->(target:User)
            WHERE c.id IN $comment_ids
            WITH
                count(c) AS total_mentions,
                sum(CASE WHEN c.offensive = 'offensive' THEN 1 ELSE 0 END) AS offensive_count,
                sum(CASE WHEN c.offensive = 'not_offensive' OR c.offensive IS NULL THEN 1 ELSE 0 END) AS non_offensive_count
            RETURN total_mentions AS outgoing_mentions,
                   offensive_count AS outgoing_offensive,
                   non_offensive_count AS outgoing_non_offensive
            """

            # 2. 被mention次数 (用户收到的总mention数量) - 基于评论集
            query2 = """
            MATCH (c:Comment)-[:MENTIONS]->(u:User {id: $userId})
            WHERE c.id IN $comment_ids
            WITH
                count(c) AS total_mentions,
                sum(CASE WHEN c.offensive = 'offensive' THEN 1 ELSE 0 END) AS offensive_count,
                sum(CASE WHEN c.offensive = 'not_offensive' OR c.offensive IS NULL THEN 1 ELSE 0 END) AS non_offensive_count
            RETURN total_mentions AS incoming_mentions,
                   offensive_count AS incoming_offensive,
                   non_offensive_count AS incoming_non_offensive
            """

            # 8. 被mention的用户数量 (该用户被多少个不同的用户mention过)
            # 9. mention的用户数量 (该用户mention过多少个不同的用户)
            query3 = """
            MATCH (u:User {id: $userId})-[:CREATES]->(c:Comment)-[:MENTIONS]->(target:User)
            WHERE c.id IN $comment_ids
            RETURN count(DISTINCT target) AS unique_targets
            """

            query4 = """
            MATCH (c:Comment)-[:MENTIONS]->(u:User {id: $userId})
            WHERE c.id IN $comment_ids
            OPTIONAL MATCH (source:User)-[:CREATES]->(c)
            RETURN count(DISTINCT source) AS unique_sources
            """

            # 10. 对用户mention的最大次数 (该用户对单一目标用户mention的最大次数，不区分offensive/non-offensive)
            # 11. 对一位用户的最大OFFENSIVE次数 (该用户对单一目标用户发出OFFENSIVE mention的最大次数)
            # 12. 对一位用户的最大NON_OFFENSIVE次数 (该用户对单一目标用户发出NON_OFFENSIVE mention的最大次数)
            query5 = """
            MATCH (u:User {id: $userId})-[:CREATES]->(c:Comment)-[:MENTIONS]->(target:User)
            WHERE c.id IN $comment_ids
            WITH target,
                 count(c) AS total_mentions,
                 sum(CASE WHEN c.offensive = 'offensive' THEN 1 ELSE 0 END) AS offensive_count,
                 sum(CASE WHEN c.offensive = 'not_offensive' OR c.offensive IS NULL THEN 1 ELSE 0 END) AS non_offensive_count
            RETURN
                COALESCE(MAX(total_mentions), 0) AS max_mentions_to_single_user,
                COALESCE(MAX(offensive_count), 0) AS max_offensive_to_single_user,
                COALESCE(MAX(non_offensive_count), 0) AS max_non_offensive_to_single_user
            """

            # 19. 被攻击集中度（收到）(从单一用户收到的最大OFFENSIVE mention次数 / 总被OFFENSIVE mention次数)
            # 20. 被NON_OFFENSIVE mention集中度（收到）(从单一用户收到的最大NON_OFFENSIVE mention次数 / 总被NON_OFFENSIVE mention次数)
            query6 = """
            MATCH (c:Comment)-[:MENTIONS]->(u:User {id: $userId})
            WHERE c.id IN $comment_ids
            OPTIONAL MATCH (source:User)-[:CREATES]->(c)
            WITH source,
                 sum(CASE WHEN c.offensive = 'offensive' THEN 1 ELSE 0 END) AS offensive_count,
                 sum(CASE WHEN c.offensive = 'not_offensive' OR c.offensive IS NULL THEN 1 ELSE 0 END) AS non_offensive_count
            RETURN
                COALESCE(MAX(offensive_count), 0) AS max_offensive_from_single_user,
                COALESCE(MAX(non_offensive_count), 0) AS max_non_offensive_from_single_user
            """

            # 21. 双向攻击用户数 (与该用户互发过（至少一次）OFFENSIVE mention的独特用户数量)
            # 22. 双向NON_OFFENSIVE互动用户数 (与该用户互发过（至少一次）NON_OFFENSIVE mention的独特用户数量)
            query7 = """
            // 基于评论集计算双向OFFENSIVE关系
            MATCH (u:User {id: $userId})
            MATCH (other:User)
            WHERE other <> u

            // 检查该用户是否向other发送过offensive mention
            WITH u, other
            WHERE EXISTS {
                MATCH (u)-[:CREATES]->(c1:Comment)-[:MENTIONS]->(other)
                WHERE c1.id IN $comment_ids AND c1.offensive = 'offensive'
            }

            // 检查other是否向该用户发送过offensive mention
            WITH u, other
            WHERE EXISTS {
                MATCH (c2:Comment)-[:MENTIONS]->(u)
                WHERE c2.id IN $comment_ids AND c2.offensive = 'offensive'
                AND EXISTS((other)-[:CREATES]->(c2))
            }

            RETURN count(DISTINCT other) AS bidirectional_offensive_users
            """

            query8 = """
            // 基于评论集计算双向NON_OFFENSIVE关系
            MATCH (u:User {id: $userId})
            MATCH (other:User)
            WHERE other <> u

            // 检查该用户是否向other发送过non_offensive mention
            WITH u, other
            WHERE EXISTS {
                MATCH (u)-[:CREATES]->(c1:Comment)-[:MENTIONS]->(other)
                WHERE c1.id IN $comment_ids AND (c1.offensive = 'not_offensive' OR c1.offensive IS NULL)
            }

            // 检查other是否向该用户发送过non_offensive mention
            WITH u, other
            WHERE EXISTS {
                MATCH (c2:Comment)-[:MENTIONS]->(u)
                WHERE c2.id IN $comment_ids AND (c2.offensive = 'not_offensive' OR c2.offensive IS NULL)
                AND EXISTS((other)-[:CREATES]->(c2))
            }

            RETURN count(DISTINCT other) AS bidirectional_non_offensive_users
            """

            # 23. 未反击比例 (衡量用户收到OFFENSIVE mention后未进行OFFENSIVE反击的程度或比例)
            query9 = """
            // 基于评论集计算未反击用户数
            MATCH (u:User {id: $userId})
            MATCH (source:User)
            WHERE source <> u

            // 找出向用户发送过OFFENSIVE评论的用户
            WITH u, source
            WHERE EXISTS {
                MATCH (c:Comment)-[:MENTIONS]->(u)
                WHERE c.id IN $comment_ids AND c.offensive = 'offensive'
                AND EXISTS((source)-[:CREATES]->(c))
            }

            // 筛选出未被该用户反击的用户
            WITH u, source
            WHERE NOT EXISTS {
                MATCH (u)-[:CREATES]->(c2:Comment)-[:MENTIONS]->(source)
                WHERE c2.id IN $comment_ids AND c2.offensive = 'offensive'
            }

            RETURN count(DISTINCT source) AS unretaliated_users
            """
        else:
            # 不限制查询范围的原始查询 - 使用MENTIONS关系
            query1 = """
            MATCH (u:User {id: $userId})-[:CREATES]->(c:Comment)-[:MENTIONS]->(:User)
            WITH
                count(c) AS total_mentions,
                sum(CASE WHEN c.offensive = 'offensive' THEN 1 ELSE 0 END) AS offensive_count,
                sum(CASE WHEN c.offensive = 'not_offensive' OR c.offensive IS NULL THEN 1 ELSE 0 END) AS non_offensive_count
            RETURN total_mentions AS outgoing_mentions,
                   offensive_count AS outgoing_offensive,
                   non_offensive_count AS outgoing_non_offensive
            """

            query2 = """
            MATCH (c:Comment)-[:MENTIONS]->(u:User {id: $userId})
            WITH
                count(c) AS total_mentions,
                sum(CASE WHEN c.offensive = 'offensive' THEN 1 ELSE 0 END) AS offensive_count,
                sum(CASE WHEN c.offensive = 'not_offensive' OR c.offensive IS NULL THEN 1 ELSE 0 END) AS non_offensive_count
            RETURN total_mentions AS incoming_mentions,
                   offensive_count AS incoming_offensive,
                   non_offensive_count AS incoming_non_offensive
            """

            # 8. 被mention的用户数量 (该用户被多少个不同的用户mention过)
            # 9. mention的用户数量 (该用户mention过多少个不同的用户)
            query3 = """
            MATCH (u:User {id: $userId})-[:CREATES]->(c:Comment)-[:MENTIONS]->(target:User)
            RETURN count(DISTINCT target) AS unique_targets
            """

            query4 = """
            MATCH (c:Comment)-[:MENTIONS]->(u:User {id: $userId})
            OPTIONAL MATCH (source:User)-[:CREATES]->(c)
            RETURN count(DISTINCT source) AS unique_sources
            """

            # 10. 对用户mention的最大次数 (该用户对单一目标用户mention的最大次数，不区分offensive/non-offensive)
            # 11. 对一位用户的最大OFFENSIVE次数 (该用户对单一目标用户发出OFFENSIVE mention的最大次数)
            # 12. 对一位用户的最大NON_OFFENSIVE次数 (该用户对单一目标用户发出NON_OFFENSIVE mention的最大次数)
            query5 = """
            MATCH (u:User {id: $userId})-[:CREATES]->(c:Comment)-[:MENTIONS]->(target:User)
            WITH target,
                 count(c) AS total_mentions,
                 sum(CASE WHEN c.offensive = 'offensive' THEN 1 ELSE 0 END) AS offensive_count,
                 sum(CASE WHEN c.offensive = 'not_offensive' OR c.offensive IS NULL THEN 1 ELSE 0 END) AS non_offensive_count
            RETURN
                COALESCE(MAX(total_mentions), 0) AS max_mentions_to_single_user,
                COALESCE(MAX(offensive_count), 0) AS max_offensive_to_single_user,
                COALESCE(MAX(non_offensive_count), 0) AS max_non_offensive_to_single_user
            """

            # 19. 被攻击集中度（收到）(从单一用户收到的最大OFFENSIVE mention次数 / 总被OFFENSIVE mention次数)
            # 20. 被NON_OFFENSIVE mention集中度（收到）(从单一用户收到的最大NON_OFFENSIVE mention次数 / 总被NON_OFFENSIVE mention次数)
            query6 = """
            MATCH (c:Comment)-[:MENTIONS]->(u:User {id: $userId})
            OPTIONAL MATCH (source:User)-[:CREATES]->(c)
            WITH source,
                 sum(CASE WHEN c.offensive = 'offensive' THEN 1 ELSE 0 END) AS offensive_count,
                 sum(CASE WHEN c.offensive = 'not_offensive' OR c.offensive IS NULL THEN 1 ELSE 0 END) AS non_offensive_count
            RETURN
                COALESCE(MAX(offensive_count), 0) AS max_offensive_from_single_user,
                COALESCE(MAX(non_offensive_count), 0) AS max_non_offensive_from_single_user
            """

            # 21. 双向攻击用户数 (与该用户互发过（至少一次）OFFENSIVE mention的独特用户数量)
            # 22. 双向NON_OFFENSIVE互动用户数 (与该用户互发过（至少一次）NON_OFFENSIVE mention的独特用户数量)
            query7 = """
            // 计算双向OFFENSIVE关系
            MATCH (u:User {id: $userId})
            MATCH (other:User)
            WHERE other <> u

            // 检查该用户是否向other发送过offensive mention
            WITH u, other
            WHERE EXISTS {
                MATCH (u)-[:CREATES]->(c1:Comment)-[:MENTIONS]->(other)
                WHERE c1.offensive = 'offensive'
            }

            // 检查other是否向该用户发送过offensive mention
            WITH u, other
            WHERE EXISTS {
                MATCH (c2:Comment)-[:MENTIONS]->(u)
                WHERE c2.offensive = 'offensive'
                AND EXISTS((other)-[:CREATES]->(c2))
            }

            RETURN count(DISTINCT other) AS bidirectional_offensive_users
            """

            query8 = """
            // 计算双向NON_OFFENSIVE关系
            MATCH (u:User {id: $userId})
            MATCH (other:User)
            WHERE other <> u

            // 检查该用户是否向other发送过non_offensive mention
            WITH u, other
            WHERE EXISTS {
                MATCH (u)-[:CREATES]->(c1:Comment)-[:MENTIONS]->(other)
                WHERE (c1.offensive = 'not_offensive' OR c1.offensive IS NULL)
            }

            // 检查other是否向该用户发送过non_offensive mention
            WITH u, other
            WHERE EXISTS {
                MATCH (c2:Comment)-[:MENTIONS]->(u)
                WHERE (c2.offensive = 'not_offensive' OR c2.offensive IS NULL)
                AND EXISTS((other)-[:CREATES]->(c2))
            }

            RETURN count(DISTINCT other) AS bidirectional_non_offensive_users
            """

            # 23. 未反击比例 (衡量用户收到OFFENSIVE mention后未进行OFFENSIVE反击的程度或比例)
            query9 = """
            // 计算未反击用户数
            MATCH (u:User {id: $userId})
            MATCH (source:User)
            WHERE source <> u

            // 找出向用户发送过OFFENSIVE评论的用户
            WITH u, source
            WHERE EXISTS {
                MATCH (c:Comment)-[:MENTIONS]->(u)
                WHERE c.offensive = 'offensive'
                AND EXISTS((source)-[:CREATES]->(c))
            }

            // 筛选出未被该用户反击的用户
            WITH u, source
            WHERE NOT EXISTS {
                MATCH (u)-[:CREATES]->(c2:Comment)-[:MENTIONS]->(source)
                WHERE c2.offensive = 'offensive'
            }

            RETURN count(DISTINCT source) AS unretaliated_users
            """

        with self.driver.session() as session:
            # 根据是否有statistic_comment_ids来决定查询参数
            if statistic_comment_ids is not None:
                params = {"userId": user_id, "comment_ids": list(statistic_comment_ids)}

                result1 = session.run(query1, params).single() or {"outgoing_mentions": 0, "outgoing_offensive": 0, "outgoing_non_offensive": 0}
                result2 = session.run(query2, params).single() or {"incoming_mentions": 0, "incoming_offensive": 0, "incoming_non_offensive": 0}
                result3 = session.run(query3, params).single() or {"unique_targets": 0}
                result4 = session.run(query4, params).single() or {"unique_sources": 0}
                result5 = session.run(query5, params).single() or {"max_mentions_to_single_user": 0, "max_offensive_to_single_user": 0, "max_non_offensive_to_single_user": 0}
                result6 = session.run(query6, params).single() or {"max_offensive_from_single_user": 0, "max_non_offensive_from_single_user": 0}
                result7 = session.run(query7, params).single() or {"bidirectional_offensive_users": 0}
                result8 = session.run(query8, params).single() or {"bidirectional_non_offensive_users": 0}
                result9 = session.run(query9, params).single() or {"unretaliated_users": 0}

                # 计算未反击比例
                incoming_offensive_query = """
                    // 基于评论集计算向用户发送过OFFENSIVE评论的用户数
                    MATCH (u:User {id: $userId})
                    MATCH (source:User)
                    WHERE source <> u

                    // 找出向用户发送过OFFENSIVE评论的用户
                    WITH u, source
                    WHERE EXISTS {
                        MATCH (c:Comment)-[:MENTIONS]->(u)
                        WHERE c.id IN $comment_ids AND c.offensive = 'offensive'
                        AND EXISTS((source)-[:CREATES]->(c))
                    }

                    RETURN count(DISTINCT source) AS count
                """
                incoming_offensive_users = session.run(incoming_offensive_query, params).single()["count"]
            else:
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
                incoming_offensive_query = """
                    // 计算向用户发送过OFFENSIVE评论的用户数
                    MATCH (u:User {id: $userId})
                    MATCH (source:User)
                    WHERE source <> u

                    // 找出向用户发送过OFFENSIVE评论的用户
                    WITH u, source
                    WHERE EXISTS {
                        MATCH (c:Comment)-[:MENTIONS]->(u)
                        WHERE c.offensive = 'offensive'
                        AND EXISTS((source)-[:CREATES]->(c))
                    }

                    RETURN count(DISTINCT source) AS count
                """
                incoming_offensive_users = session.run(incoming_offensive_query, userId=user_id).single()["count"]

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



    def get_user_comment_features(self, user_id: str, statistic_comment_ids: Set[str] = None) -> Dict[str, Any]:
        """获取用户评论相关特征

        Args:
            user_id: 用户ID
            statistic_comment_ids: 用于计算统计特征的评论ID集合，如果为None，则不限制评论范围

        Returns:
            Dict[str, Any]: 用户评论相关特征
        """
        # 根据是否有statistic_comment_ids来决定查询
        if statistic_comment_ids is not None:
            # 限制查询范围的版本 - 基于评论集
            # 1. 用户的Comment数量
            query1 = """
            MATCH (u:User {id: $userId})-[:CREATES]->(c:Comment)
            WHERE c.id IN $comment_ids
            RETURN count(c) AS comment_count
            """

            # 2. 用户的offensive Comment数量
            query2 = """
            MATCH (u:User {id: $userId})-[:CREATES]->(c:Comment)
            WHERE c.id IN $comment_ids AND c.offensive = 'offensive'
            RETURN count(c) AS offensive_comment_count
            """

            # 3. 用户的non_offensive Comment数量
            query3 = """
            MATCH (u:User {id: $userId})-[:CREATES]->(c:Comment)
            WHERE c.id IN $comment_ids AND (c.offensive = 'not_offensive' OR c.offensive IS NULL)
            RETURN count(c) AS non_offensive_comment_count
            """

            # 5. 用户在多少个不同的媒体会话中发表过评论
            query4 = """
            MATCH (u:User {id: $userId})-[:CREATES]->(c:Comment)-[:BELONGS_TO]->(m:MediaSession)
            WHERE c.id IN $comment_ids
            RETURN count(DISTINCT m) AS media_session_count
            """

            # 6. 用户平均在每个参与评论的会话中发表多少评论
            query5 = """
            MATCH (u:User {id: $userId})-[:CREATES]->(c:Comment)-[:BELONGS_TO]->(m:MediaSession)
            WHERE c.id IN $comment_ids
            WITH m, count(c) AS comments_per_session
            RETURN avg(comments_per_session) AS avg_comments_per_session
            """

            # 7. 用户平均在每个参与评论的会话中发表多少侵略性评论
            query6 = """
            MATCH (u:User {id: $userId})-[:CREATES]->(c:Comment)-[:BELONGS_TO]->(m:MediaSession)
            WHERE c.id IN $comment_ids AND c.offensive = 'offensive'
            WITH m, count(c) AS offensive_comments_per_session
            RETURN avg(offensive_comments_per_session) AS avg_offensive_comments_per_session
            """

            # 8. 用户发表过至少一条侵略性评论的会话占其所有评论会话的比例
            query7 = """
            MATCH (u:User {id: $userId})-[:CREATES]->(c:Comment)-[:BELONGS_TO]->(m:MediaSession)
            WHERE c.id IN $comment_ids
            WITH m, collect(c.offensive) AS offensives
            WHERE 'offensive' IN offensives
            RETURN count(DISTINCT m) AS offensive_sessions_count
            """

            # 9. 用户发表侵略性评论的频率（单位：评论数/天数）
            query8 = """
            MATCH (u:User {id: $userId})-[:CREATES]->(c:Comment)
            WHERE c.id IN $comment_ids AND c.offensive = 'offensive' AND c.created IS NOT NULL
            WITH min(c.created) AS first_date, max(c.created) AS last_date, count(c) AS offensive_count
            RETURN
                CASE
                    WHEN duration.inDays(datetime(first_date), datetime(last_date)).days > 0
                    THEN toFloat(offensive_count) / duration.inDays(datetime(first_date), datetime(last_date)).days
                    ELSE toFloat(offensive_count)
                END AS offensive_frequency
            """

            # 10. 用户在会话中的第一条评论是侵略性的会话比例
            query9 = """
            MATCH (u:User {id: $userId})-[:CREATES]->(c:Comment)-[:BELONGS_TO]->(m:MediaSession)
            WHERE c.id IN $comment_ids
            WITH m, c ORDER BY c.created
            WITH m, collect(c)[0] AS first_comment
            WHERE first_comment.offensive = 'offensive'
            RETURN count(DISTINCT m) AS first_comment_offensive_count
            """

            # 11. 用户在会话中的最后一条评论是侵略性的会话比例
            query10 = """
            MATCH (u:User {id: $userId})-[:CREATES]->(c:Comment)-[:BELONGS_TO]->(m:MediaSession)
            WHERE c.id IN $comment_ids
            WITH m, c ORDER BY c.created DESC
            WITH m, collect(c)[0] AS last_comment
            WHERE last_comment.offensive = 'offensive'
            RETURN count(DISTINCT m) AS last_comment_offensive_count
            """
        else:
            # 不限制查询范围的原始查询
            # 1. 用户的Comment数量
            query1 = """
            MATCH (u:User {id: $userId})-[:CREATES]->(c:Comment)
            RETURN count(c) AS comment_count
            """

            # 2. 用户的offensive Comment数量
            query2 = """
            MATCH (u:User {id: $userId})-[:CREATES]->(c:Comment)
            WHERE c.offensive = 'offensive'
            RETURN count(c) AS offensive_comment_count
            """

            # 3. 用户的non_offensive Comment数量
            query3 = """
            MATCH (u:User {id: $userId})-[:CREATES]->(c:Comment)
            WHERE c.offensive = 'not_offensive' OR c.offensive IS NULL
            RETURN count(c) AS non_offensive_comment_count
            """

            # 5. 用户在多少个不同的媒体会话中发表过评论
            query4 = """
            MATCH (u:User {id: $userId})-[:CREATES]->(c:Comment)-[:BELONGS_TO]->(m:MediaSession)
            RETURN count(DISTINCT m) AS media_session_count
            """

            # 6. 用户平均在每个参与评论的会话中发表多少评论
            query5 = """
            MATCH (u:User {id: $userId})-[:CREATES]->(c:Comment)-[:BELONGS_TO]->(m:MediaSession)
            WITH m, count(c) AS comments_per_session
            RETURN avg(comments_per_session) AS avg_comments_per_session
            """

            # 7. 用户平均在每个参与评论的会话中发表多少侵略性评论
            query6 = """
            MATCH (u:User {id: $userId})-[:CREATES]->(c:Comment)-[:BELONGS_TO]->(m:MediaSession)
            WHERE c.offensive = 'offensive'
            WITH m, count(c) AS offensive_comments_per_session
            RETURN avg(offensive_comments_per_session) AS avg_offensive_comments_per_session
            """

            # 8. 用户发表过至少一条侵略性评论的会话占其所有评论会话的比例
            query7 = """
            MATCH (u:User {id: $userId})-[:CREATES]->(c:Comment)-[:BELONGS_TO]->(m:MediaSession)
            WITH m, collect(c.offensive) AS offensives
            WHERE 'offensive' IN offensives
            RETURN count(DISTINCT m) AS offensive_sessions_count
            """

            # 9. 用户发表侵略性评论的频率（单位：评论数/天数）
            query8 = """
            MATCH (u:User {id: $userId})-[:CREATES]->(c:Comment)
            WHERE c.offensive = 'offensive' AND c.created IS NOT NULL
            WITH min(c.created) AS first_date, max(c.created) AS last_date, count(c) AS offensive_count
            RETURN
                CASE
                    WHEN duration.inDays(datetime(first_date), datetime(last_date)).days > 0
                    THEN toFloat(offensive_count) / duration.inDays(datetime(first_date), datetime(last_date)).days
                    ELSE toFloat(offensive_count)
                END AS offensive_frequency
            """

            # 10. 用户在会话中的第一条评论是侵略性的会话比例
            query9 = """
            MATCH (u:User {id: $userId})-[:CREATES]->(c:Comment)-[:BELONGS_TO]->(m:MediaSession)
            WITH m, c ORDER BY c.created
            WITH m, collect(c)[0] AS first_comment
            WHERE first_comment.offensive = 'offensive'
            RETURN count(DISTINCT m) AS first_comment_offensive_count
            """

            # 11. 用户在会话中的最后一条评论是侵略性的会话比例
            query10 = """
            MATCH (u:User {id: $userId})-[:CREATES]->(c:Comment)-[:BELONGS_TO]->(m:MediaSession)
            WITH m, c ORDER BY c.created DESC
            WITH m, collect(c)[0] AS last_comment
            WHERE last_comment.offensive = 'offensive'
            RETURN count(DISTINCT m) AS last_comment_offensive_count
            """

        with self.driver.session() as session:
            # 根据是否有statistic_comment_ids来决定查询参数
            if statistic_comment_ids is not None:
                params = {"userId": user_id, "comment_ids": list(statistic_comment_ids)}

                result1 = session.run(query1, params).single() or {"comment_count": 0}
                result2 = session.run(query2, params).single() or {"offensive_comment_count": 0}
                result3 = session.run(query3, params).single() or {"non_offensive_comment_count": 0}
                result4 = session.run(query4, params).single() or {"media_session_count": 0}
                result5 = session.run(query5, params).single() or {"avg_comments_per_session": 0}
                result6 = session.run(query6, params).single() or {"avg_offensive_comments_per_session": 0}
                result7 = session.run(query7, params).single() or {"offensive_sessions_count": 0}
                result8 = session.run(query8, params).single() or {"offensive_frequency": 0}
                result9 = session.run(query9, params).single() or {"first_comment_offensive_count": 0}
                result10 = session.run(query10, params).single() or {"last_comment_offensive_count": 0}
            else:
                result1 = session.run(query1, userId=user_id).single() or {"comment_count": 0}
                result2 = session.run(query2, userId=user_id).single() or {"offensive_comment_count": 0}
                result3 = session.run(query3, userId=user_id).single() or {"non_offensive_comment_count": 0}
                result4 = session.run(query4, userId=user_id).single() or {"media_session_count": 0}
                result5 = session.run(query5, userId=user_id).single() or {"avg_comments_per_session": 0}
                result6 = session.run(query6, userId=user_id).single() or {"avg_offensive_comments_per_session": 0}
                result7 = session.run(query7, userId=user_id).single() or {"offensive_sessions_count": 0}
                result8 = session.run(query8, userId=user_id).single() or {"offensive_frequency": 0}
                result9 = session.run(query9, userId=user_id).single() or {"first_comment_offensive_count": 0}
                result10 = session.run(query10, userId=user_id).single() or {"last_comment_offensive_count": 0}

            # 计算比例
            comment_count = result1["comment_count"]
            media_session_count = result4["media_session_count"]
            offensive_sessions_count = result7["offensive_sessions_count"]

            # 4. 用户的侵略性评论比例
            offensive_ratio = 0
            if comment_count > 0:
                offensive_ratio = result2["offensive_comment_count"] / comment_count

            # 8. 用户发表过至少一条侵略性评论的会话占比
            offensive_sessions_ratio = 0
            if media_session_count > 0:
                offensive_sessions_ratio = offensive_sessions_count / media_session_count

            # 10. 第一条评论是侵略性的会话比例
            first_comment_offensive_ratio = 0
            if media_session_count > 0:
                first_comment_offensive_ratio = result9["first_comment_offensive_count"] / media_session_count

            # 11. 最后一条评论是侵略性的会话比例
            last_comment_offensive_ratio = 0
            if media_session_count > 0:
                last_comment_offensive_ratio = result10["last_comment_offensive_count"] / media_session_count

            return {
                "comment_count": comment_count,
                "offensive_comment_count": result2["offensive_comment_count"],
                "non_offensive_comment_count": result3["non_offensive_comment_count"],
                "offensive_comment_ratio": offensive_ratio,
                "media_session_count": media_session_count,
                "avg_comments_per_session": result5["avg_comments_per_session"],
                "avg_offensive_comments_per_session": result6["avg_offensive_comments_per_session"],
                "offensive_sessions_ratio": offensive_sessions_ratio,
                "offensive_comment_frequency": result8["offensive_frequency"],
                "first_comment_offensive_ratio": first_comment_offensive_ratio,
                "last_comment_offensive_ratio": last_comment_offensive_ratio
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

def split_media_sessions_by_time(media_sessions: List[Tuple[str, str]], train_ratio: float = 0.8, val_ratio: float = 0.1) -> Tuple[List[str], List[str], List[str]]:
    """按时间顺序划分媒体会话

    Args:
        media_sessions: 媒体会话ID和创建时间的元组列表，按时间排序
        train_ratio: 训练集比例
        val_ratio: 验证集比例

    Returns:
        Tuple[List[str], List[str], List[str]]: 训练集、验证集和测试集的媒体会话ID列表
    """
    # 确保media_sessions按时间排序
    sorted_sessions = sorted(media_sessions, key=lambda x: x[1])

    # 计算划分点
    n = len(sorted_sessions)
    train_size = int(n * train_ratio)
    val_size = int(n * val_ratio)

    # 划分数据集
    train_sessions = [session[0] for session in sorted_sessions[:train_size]]
    val_sessions = [session[0] for session in sorted_sessions[train_size:train_size+val_size]]
    test_sessions = [session[0] for session in sorted_sessions[train_size+val_size:]]

    return train_sessions, val_sessions, test_sessions

def extract_features_for_split(extractor: UserFeaturesExtractor, split_name: str, target_users: List[str], statistic_media_sessions: List[str], output_path: str):
    """为特定数据集划分提取特征

    Args:
        extractor: 特征提取器
        split_name: 划分名称（train, val, test）
        target_users: 目标用户列表
        statistic_media_sessions: 用于统计的媒体会话列表
        output_path: 输出文件路径
    """
    print(f"为{split_name}集提取特征...")
    print(f"目标用户数量: {len(target_users)}")
    print(f"统计媒体会话数量: {len(statistic_media_sessions)}")

    # 提取特征
    user_features = extractor.extract_user_features(target_users, statistic_media_sessions)

    # 保存特征
    extractor.save_features_to_csv(user_features, output_path)

    print(f"{split_name}集特征已保存到 {output_path}")

def main():
    """主函数"""
    # Neo4j连接信息
    uri = "bolt://localhost:7687"
    username = "neo4j"
    password = "12345678"  # 请替换为实际密码

    # 输出目录
    output_dir = "data/processed/features_extended"
    os.makedirs(output_dir, exist_ok=True)

    # 初始化特征提取器
    extractor = UserFeaturesExtractor(uri, username, password)

    try:
        # 按时间顺序获取媒体会话
        media_sessions = extractor.get_media_sessions_by_time()

        # 按时间顺序划分媒体会话
        train_sessions, val_sessions, test_sessions = split_media_sessions_by_time(media_sessions)

        print(f"训练集大小: {len(train_sessions)}")
        print(f"验证集大小: {len(val_sessions)}")
        print(f"测试集大小: {len(test_sessions)}")

        # 获取各个划分中的用户
        train_users = extractor.get_users_in_media_sessions(train_sessions)
        val_users = extractor.get_users_in_media_sessions(val_sessions)
        test_users = extractor.get_users_in_media_sessions(test_sessions)

        print(f"训练集用户数: {len(train_users)}")
        print(f"验证集用户数: {len(val_users)}")
        print(f"测试集用户数: {len(test_users)}")

        # 为各个划分提取特征
        # 训练集特征 - 只使用训练集数据
        extract_features_for_split(
            extractor=extractor,
            split_name="训练",
            target_users=train_users,
            statistic_media_sessions=train_sessions,
            output_path=f"{output_dir}/train_features.csv"
        )

        # 验证集特征 - 使用训练集+验证集数据
        extract_features_for_split(
            extractor=extractor,
            split_name="验证",
            target_users=val_users,
            statistic_media_sessions=train_sessions + val_sessions,
            output_path=f"{output_dir}/val_features.csv"
        )

        # 测试集特征 - 使用所有数据
        extract_features_for_split(
            extractor=extractor,
            split_name="测试",
            target_users=test_users,
            statistic_media_sessions=train_sessions + val_sessions + test_sessions,
            output_path=f"{output_dir}/test_features.csv"
        )

        # 保存数据集划分
        splits_dir = "data/processed/splits"
        os.makedirs(splits_dir, exist_ok=True)

        with open(f"{splits_dir}/train_sessions.json", "w") as f:
            json.dump(train_sessions, f)

        with open(f"{splits_dir}/val_sessions.json", "w") as f:
            json.dump(val_sessions, f)

        with open(f"{splits_dir}/test_sessions.json", "w") as f:
            json.dump(test_sessions, f)

        with open(f"{splits_dir}/train_users.json", "w") as f:
            json.dump(train_users, f)

        with open(f"{splits_dir}/val_users.json", "w") as f:
            json.dump(val_users, f)

        with open(f"{splits_dir}/test_users.json", "w") as f:
            json.dump(test_users, f)

        print("数据集划分已保存")

    finally:
        # 关闭连接
        extractor.close()

if __name__ == "__main__":
    main()