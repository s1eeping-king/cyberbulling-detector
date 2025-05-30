import os
import numpy as np
import pandas as pd
from neo4j import GraphDatabase
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Set
import logging
import json

class BullyingDetector:
    """使用随机森林模型进行霸凌检测的分类器"""

    def __init__(self, uri: str, username: str, password: str):
        """初始化霸凌检测器

        Args:
            uri: Neo4j数据库URI
            username: 数据库用户名
            password: 数据库密码
        """
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
        self.logger = self.setup_logger()

        # 确保输出目录存在
        os.makedirs("experiments_others/v2/results", exist_ok=True)

    @staticmethod
    def setup_logger():
        """设置日志记录器"""
        logger = logging.getLogger('BullyingDetector')
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def close(self):
        """关闭Neo4j驱动连接"""
        self.driver.close()

    def get_media_sessions_with_users(self) -> Dict[str, Dict]:
        """获取所有媒体会话及其关联的用户

        Returns:
            Dict[str, Dict]: 媒体会话ID到会话信息的映射
        """
        # 修改查询，确保获取所有媒体会话，不仅仅是有标签的
        query = """
        MATCH (m:MediaSession)
        OPTIONAL MATCH (m)-[:HAS_LABEL]->(l:Label)
        WITH m, l.value AS label
        MATCH (u:User)-[:PARTICIPATES_IN]->(m)
        WITH m, label, collect(DISTINCT u.id) AS user_ids
        WHERE size(user_ids) > 0
        RETURN m.id AS media_id, label, user_ids
        """

        with self.driver.session() as session:
            result = session.run(query)
            media_sessions = {}
            bullying_count = 0
            non_bullying_count = 0

            for record in result:
                media_id = record["media_id"]
                label = record["label"]
                user_ids = record["user_ids"]

                # 只保留有用户且有标签的会话
                if user_ids and label is not None:
                    is_bullying = label == "bullying"
                    # 确保用户ID作为字符串存储
                    user_ids_str = [str(uid) for uid in user_ids]
                    media_sessions[media_id] = {
                        "label": 1 if is_bullying else 0,
                        "user_ids": user_ids_str
                    }

                    if is_bullying:
                        bullying_count += 1
                    else:
                        non_bullying_count += 1

            self.logger.info(f"找到 {len(media_sessions)} 个有用户的媒体会话，其中霸凌会话 {bullying_count} 个，非霸凌会话 {non_bullying_count} 个")
            return media_sessions

    def load_user_features(self) -> Dict[str, Dict]:
        """加载用户特征

        Returns:
            Dict[str, Dict]: 用户ID到特征的映射
        """
        # 从CSV文件加载用户特征
        features_path = "data/processed/features_extended/user_features_extended.csv"

        if not os.path.exists(features_path):
            self.logger.error(f"用户特征文件 {features_path} 不存在")
            return {}

        # 读取CSV文件时将userId列指定为字符串类型，避免科学计数法导致的精度丢失
        df = pd.read_csv(features_path, dtype={'userId': str})

        # 将DataFrame转换为字典，确保userId作为字符串存储
        user_features = {}
        for _, row in df.iterrows():
            # 确保userId作为字符串存储，避免精度丢失
            user_id = str(row["userId"])
            # 删除userId列，只保留特征
            features = row.drop("userId").to_dict()
            user_features[user_id] = features

        self.logger.info(f"加载了 {len(user_features)} 个用户的特征")
        return user_features

    def aggregate_features_for_media_sessions(self, media_sessions: Dict[str, Dict], user_features: Dict[str, Dict]) -> Tuple[pd.DataFrame, pd.Series]:
        """为每个媒体会话聚合用户特征

        Args:
            media_sessions: 媒体会话ID到会话信息的映射
            user_features: 用户ID到特征的映射

        Returns:
            Tuple[pd.DataFrame, pd.Series]: 特征矩阵和标签向量
        """
        aggregated_features = []
        labels = []
        media_ids = []

        # 统计信息
        total_sessions = len(media_sessions)
        sessions_with_features = 0
        sessions_without_features = 0
        total_users = 0
        users_with_features = 0
        users_without_features = 0

        for media_id, session_info in media_sessions.items():
            label = session_info["label"]
            user_ids = session_info["user_ids"]
            total_users += len(user_ids)

            # 获取该会话中所有用户的特征
            session_user_features = []
            session_users_with_features = 0
            session_users_without_features = 0

            for user_id in user_ids:
                if user_id in user_features:
                    session_user_features.append(user_features[user_id])
                    session_users_with_features += 1
                    users_with_features += 1
                else:
                    session_users_without_features += 1
                    users_without_features += 1

            # 如果没有用户特征，跳过该会话
            if not session_user_features:
                sessions_without_features += 1
                continue

            sessions_with_features += 1

            # 计算每个特征的平均值
            avg_features = {}
            for feature_name in session_user_features[0].keys():
                # 收集所有用户的该特征值
                feature_values = [user_feature.get(feature_name, 0) for user_feature in session_user_features]
                # 计算平均值
                avg_features[feature_name] = sum(feature_values) / len(feature_values)

            aggregated_features.append(avg_features)
            labels.append(label)
            media_ids.append(media_id)

        # 转换为DataFrame和Series
        X = pd.DataFrame(aggregated_features)
        y = pd.Series(labels)

        # 打印统计信息
        self.logger.info(f"总媒体会话数: {total_sessions}")
        self.logger.info(f"有用户特征的会话数: {sessions_with_features}")
        self.logger.info(f"没有用户特征的会话数: {sessions_without_features}")
        self.logger.info(f"总用户数: {total_users}")
        self.logger.info(f"有特征的用户数: {users_with_features}")
        self.logger.info(f"没有特征的用户数: {users_without_features}")
        self.logger.info(f"聚合了 {len(aggregated_features)} 个媒体会话的特征")

        # 检查是否有足够的数据进行训练
        if len(aggregated_features) == 0:
            self.logger.error("没有足够的数据进行训练，请检查用户特征和媒体会话数据")

        # 检查标签分布
        if len(labels) > 0:
            bullying_count = sum(labels)
            non_bullying_count = len(labels) - bullying_count
            self.logger.info(f"标签分布: 霸凌 {bullying_count}, 非霸凌 {non_bullying_count}")

        return X, y, media_ids

    def train_and_evaluate(self, X: pd.DataFrame, y: pd.Series) -> RandomForestClassifier:
        """训练和评估随机森林模型

        Args:
            X: 特征矩阵
            y: 标签向量

        Returns:
            RandomForestClassifier: 训练好的模型
        """
        # 检查数据集大小
        if len(X) < 10:
            self.logger.warning(f"数据集太小 ({len(X)} 个样本)，可能无法进行有效的训练和评估")

        # 划分训练集和测试集
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        except ValueError as e:
            self.logger.error(f"划分数据集时出错: {e}")
            # 如果stratify失败，尝试不使用stratify
            self.logger.info("尝试不使用stratify进行划分")
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # 标准化特征
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # 设置随机森林参数网格
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }

        # 使用交叉验证进行网格搜索
        cv = StratifiedKFold(n_splits=min(5, len(X_train)), shuffle=True, random_state=42)
        grid_search = GridSearchCV(
            RandomForestClassifier(random_state=42, class_weight='balanced'),
            param_grid=param_grid,
            cv=cv,
            scoring='f1',
            n_jobs=-1
        )

        # 训练模型
        self.logger.info("开始训练随机森林模型...")
        grid_search.fit(X_train_scaled, y_train)

        # 获取最佳模型
        best_model = grid_search.best_estimator_
        self.logger.info(f"最佳参数: {grid_search.best_params_}")

        # 在测试集上评估模型
        y_pred = best_model.predict(X_test_scaled)

        # 计算评估指标
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        self.logger.info(f"准确率: {accuracy:.4f}")
        self.logger.info(f"精确率: {precision:.4f}")
        self.logger.info(f"召回率: {recall:.4f}")
        self.logger.info(f"F1分数: {f1:.4f}")

        # 打印分类报告
        self.logger.info("\n分类报告:")
        self.logger.info(classification_report(y_test, y_pred, zero_division=0))

        # 绘制混淆矩阵
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('预测标签')
        plt.ylabel('真实标签')
        plt.title('混淆矩阵')
        plt.savefig('experiments_others/v2/results/confusion_matrix.png')

        # 特征重要性
        feature_importances = best_model.feature_importances_
        feature_names = X.columns

        # 按重要性排序
        indices = np.argsort(feature_importances)[::-1]

        # 保存特征重要性
        with open('experiments_others/v2/results/feature_importances.json', 'w') as f:
            json.dump({feature_names[i]: float(feature_importances[i]) for i in indices}, f, indent=4)

        # 绘制特征重要性
        plt.figure(figsize=(12, 8))
        plt.title('特征重要性')
        plt.bar(range(X.shape[1]), feature_importances[indices], align='center')
        plt.xticks(range(X.shape[1]), [feature_names[i] for i in indices], rotation=90)
        plt.tight_layout()
        plt.savefig('experiments_others/v2/results/feature_importances.png')

        return best_model

    def run(self):
        """运行霸凌检测器"""
        try:
            # 获取媒体会话及其关联的用户
            media_sessions = self.get_media_sessions_with_users()

            # 加载用户特征
            user_features = self.load_user_features()

            # 为每个媒体会话聚合用户特征
            X, y, media_ids = self.aggregate_features_for_media_sessions(media_sessions, user_features)

            # 检查是否有足够的数据进行训练
            if len(X) == 0:
                self.logger.error("没有足够的数据进行训练，请检查用户特征和媒体会话数据")
                return

            # 保存聚合后的特征和标签
            aggregated_data = X.copy()
            aggregated_data['label'] = y
            aggregated_data['media_id'] = media_ids
            aggregated_data.to_csv('experiments_others/v2/results/aggregated_features.csv', index=False)

            # 检查标签分布
            bullying_count = sum(y)
            non_bullying_count = len(y) - bullying_count

            if bullying_count == 0 or non_bullying_count == 0:
                self.logger.error(f"标签分布不平衡: 霸凌 {bullying_count}, 非霸凌 {non_bullying_count}")
                self.logger.error("无法训练模型，需要至少有一个霸凌样本和一个非霸凌样本")
                return

            # 训练和评估模型
            model = self.train_and_evaluate(X, y)

            self.logger.info("霸凌检测器运行完成")

        finally:
            # 关闭连接
            self.close()

def main():
    """主函数"""
    # Neo4j连接信息
    uri = "bolt://localhost:7687"
    username = "neo4j"
    password = "12345678"  # 请替换为实际密码

    # 初始化霸凌检测器
    detector = BullyingDetector(uri, username, password)

    # 运行检测器
    detector.run()

if __name__ == "__main__":
    main()
