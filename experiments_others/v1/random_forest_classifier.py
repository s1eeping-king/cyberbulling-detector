import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
from typing import Tuple, Dict, List, Optional
import time

from rf_data_loader import RFDataLoader

class CyberbullyingRandomForestClassifier:
    """使用随机森林分类器检测网络霸凌的模型"""

    def __init__(self, n_estimators: int = 100, max_depth: int = None,
                 min_samples_split: int = 2, min_samples_leaf: int = 1,
                 random_state: int = 42):
        """初始化随机森林分类器

        Args:
            n_estimators: 森林中树的数量
            max_depth: 树的最大深度
            min_samples_split: 分裂内部节点所需的最小样本数
            min_samples_leaf: 叶节点所需的最小样本数
            random_state: 随机种子
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state

        # 初始化模型
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            class_weight='balanced'  # 处理类别不平衡
        )

        # 初始化数据预处理器
        self.scaler = StandardScaler()

        # 特征名称
        self.feature_names = None

    def preprocess_data(self, X: np.ndarray) -> np.ndarray:
        """预处理数据

        Args:
            X: 特征矩阵

        Returns:
            np.ndarray: 预处理后的特征矩阵
        """
        # 标准化
        X_scaled = self.scaler.transform(X)
        return X_scaled

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: Optional[List[str]] = None) -> None:
        """训练模型

        Args:
            X: 特征矩阵
            y: 标签向量
            feature_names: 特征名称列表
        """
        print(f"训练数据形状: X={X.shape}, y={y.shape}")
        print(f"类别分布: {np.bincount(y.astype(int))}")

        # 保存特征名称
        self.feature_names = feature_names

        # 拟合标准化器
        self.scaler.fit(X)

        # 标准化数据
        X_scaled = self.scaler.transform(X)

        # 训练模型
        start_time = time.time()
        self.model.fit(X_scaled, y)
        training_time = time.time() - start_time
        print(f"模型训练完成，耗时: {training_time:.2f}秒")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测

        Args:
            X: 特征矩阵

        Returns:
            np.ndarray: 预测标签
        """
        X_processed = self.preprocess_data(X)
        return self.model.predict(X_processed)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """预测概率

        Args:
            X: 特征矩阵

        Returns:
            np.ndarray: 预测概率
        """
        X_processed = self.preprocess_data(X)
        return self.model.predict_proba(X_processed)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """评估模型

        Args:
            X: 特征矩阵
            y: 标签向量

        Returns:
            Dict: 评估指标
        """
        X_processed = self.preprocess_data(X)
        y_pred = self.model.predict(X_processed)

        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred, average='binary')
        recall = recall_score(y, y_pred, average='binary')
        f1 = f1_score(y, y_pred, average='binary')

        print(f"准确率: {accuracy:.4f}")
        print(f"精确率: {precision:.4f}")
        print(f"召回率: {recall:.4f}")
        print(f"F1分数: {f1:.4f}")

        print("\n分类报告:")
        print(classification_report(y, y_pred))

        print("\n混淆矩阵:")
        cm = confusion_matrix(y, y_pred)
        print(cm)

        # 创建输出目录
        os.makedirs('experiments_others/v1_randomForest/outputs', exist_ok=True)

        # 绘制混淆矩阵
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('预测标签')
        plt.ylabel('真实标签')
        plt.title('混淆矩阵')
        plt.savefig('experiments_others/v1_randomForest/outputs/confusion_matrix.png')
        plt.close()

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm
        }

    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """获取特征重要性

        Args:
            top_n: 返回前N个重要特征

        Returns:
            pd.DataFrame: 特征重要性DataFrame
        """
        # 确保特征名称和特征重要性长度一致
        feature_importances = self.model.feature_importances_

        if self.feature_names is None or len(self.feature_names) != len(feature_importances):
            # 如果特征名称为空或长度不匹配，创建默认名称
            feature_names = [f"feature_{i}" for i in range(len(feature_importances))]
        else:
            feature_names = self.feature_names

        # 创建特征重要性DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': feature_importances
        })

        # 按重要性排序
        importance_df = importance_df.sort_values('importance', ascending=False).reset_index(drop=True)

        # 创建输出目录
        os.makedirs('experiments_others/v1_randomForest/outputs', exist_ok=True)

        # 绘制特征重要性
        plt.figure(figsize=(12, 8))
        sns.barplot(x='importance', y='feature', data=importance_df.head(top_n))
        plt.title(f'Top {top_n} 特征重要性')
        plt.tight_layout()
        plt.savefig('experiments_others/v1_randomForest/outputs/feature_importance.png')
        plt.close()

        return importance_df

    def grid_search(self, X: np.ndarray, y: np.ndarray, param_grid: Dict) -> None:
        """网格搜索优化超参数

        Args:
            X: 特征矩阵
            y: 标签向量
            param_grid: 参数网格
        """
        X_processed = self.preprocess_data(X)

        # 创建网格搜索
        grid_search = GridSearchCV(
            estimator=RandomForestClassifier(random_state=self.random_state, class_weight='balanced'),
            param_grid=param_grid,
            cv=5,
            scoring='f1',
            n_jobs=-1
        )

        # 执行网格搜索
        print("开始网格搜索...")
        start_time = time.time()
        grid_search.fit(X_processed, y)
        search_time = time.time() - start_time

        print(f"网格搜索完成，耗时: {search_time:.2f}秒")
        print(f"最佳参数: {grid_search.best_params_}")
        print(f"最佳F1分数: {grid_search.best_score_:.4f}")

        # 更新模型参数
        self.model = grid_search.best_estimator_

        # 如果最佳参数包含n_estimators，更新实例变量
        if 'n_estimators' in param_grid:
            self.n_estimators = grid_search.best_params_.get('n_estimators', self.n_estimators)

        # 如果最佳参数包含max_depth，更新实例变量
        if 'max_depth' in param_grid:
            self.max_depth = grid_search.best_params_.get('max_depth', self.max_depth)

        # 如果最佳参数包含min_samples_split，更新实例变量
        if 'min_samples_split' in param_grid:
            self.min_samples_split = grid_search.best_params_.get('min_samples_split', self.min_samples_split)

        # 如果最佳参数包含min_samples_leaf，更新实例变量
        if 'min_samples_leaf' in param_grid:
            self.min_samples_leaf = grid_search.best_params_.get('min_samples_leaf', self.min_samples_leaf)

    def save_model(self, model_path: str) -> None:
        """保存模型

        Args:
            model_path: 模型保存路径
        """
        # 创建目录
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        # 保存模型
        joblib.dump(self, model_path)
        print(f"模型已保存到: {model_path}")

    @classmethod
    def load_model(cls, model_path: str) -> 'CyberbullyingRandomForestClassifier':
        """加载模型

        Args:
            model_path: 模型路径

        Returns:
            CyberbullyingRandomForestClassifier: 加载的模型
        """
        model = joblib.load(model_path)
        print(f"模型已从 {model_path} 加载")
        return model


def main():
    """主函数"""
    # 连接Neo4j数据库
    uri = "bolt://localhost:7687"
    user = "neo4j"
    password = "12345678"  # 替换为实际密码

    # 初始化数据加载器
    data_loader = RFDataLoader(uri=uri, user=user, password=password)

    # 加载数据集
    print("加载数据集...")
    X, y = data_loader.load_dataset()

    if X.size == 0 or y.size == 0:
        print("错误: 未能加载数据集")
        return

    print(f"数据集形状: X={X.shape}, y={y.shape}")
    print(f"类别分布: {np.bincount(y.astype(int))}")

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 获取特征名称
    feature_names = []
    for i in range(X.shape[1]):
        if i < 13:  # 前13个特征是手工特征
            feature_names.append([
                "like_count", "comment_count", "loop_count", "repost_count",
                "description_sentiment", "description_sentiment_confidence",
                "avg_follower_count", "avg_following_count", "avg_post_count",
                "avg_comment_sentiment", "positive_comment_ratio", "neutral_comment_ratio",
                "negative_comment_ratio"
            ][i])
        else:
            feature_names.append(f"bert_{i-13}")

    # 初始化随机森林分类器
    rf_classifier = CyberbullyingRandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42
    )

    # 训练模型
    print("训练模型...")
    rf_classifier.fit(X_train, y_train, feature_names)

    # 评估模型
    print("评估模型...")
    metrics = rf_classifier.evaluate(X_test, y_test)

    # 获取特征重要性
    print("计算特征重要性...")
    importance_df = rf_classifier.get_feature_importance(top_n=20)
    print(importance_df.head(20))

    # 网格搜索优化超参数
    print("执行网格搜索...")
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    rf_classifier.grid_search(X_train, y_train, param_grid)

    # 使用优化后的模型重新评估
    print("使用优化后的模型重新评估...")
    metrics = rf_classifier.evaluate(X_test, y_test)

    # 创建模型保存目录
    os.makedirs("experiments_others/v1_randomForest/models", exist_ok=True)

    # 保存模型
    rf_classifier.save_model("experiments_others/v1_randomForest/models/rf_cyberbullying_classifier.joblib")

    # 关闭数据库连接
    data_loader.close()


if __name__ == "__main__":
    main()
