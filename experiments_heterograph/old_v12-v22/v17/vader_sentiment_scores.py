import json
import os
import numpy as np
from tqdm import tqdm
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


class VaderSentimentAnalyzer:
    """VADER情感分析器，用于从文本中提取情感分数"""

    def __init__(self):
        """
        初始化VADER情感分析器
        """
        print("初始化VADER情感分析器...")
        self.analyzer = SentimentIntensityAnalyzer()
        print("VADER情感分析器初始化完成")

    def analyze_sentiment(self, text):
        """
        分析单个文本的情感

        Args:
            text: 输入文本

        Returns:
            dict: 包含情感分数的字典
                - compound: 综合情感分数 (-1到1，负数表示负面，正数表示正面)
                - pos: 正面情感分数 (0到1)
                - neu: 中性情感分数 (0到1)
                - neg: 负面情感分数 (0到1)
        """
        # 处理空文本
        if not text or text.isspace():
            return {
                'compound': 0.0,
                'pos': 0.0,
                'neu': 1.0,
                'neg': 0.0
            }

        # 使用VADER分析情感
        scores = self.analyzer.polarity_scores(text)
        return scores

    def analyze_texts(self, texts):
        """
        批量分析文本情感

        Args:
            texts: 文本列表

        Returns:
            list: 情感分数列表，每个元素是包含情感分数的字典
        """
        results = []

        for text in tqdm(texts, desc="分析文本情感"):
            sentiment_scores = self.analyze_sentiment(text)
            results.append(sentiment_scores)

        return results

    def process_comments(self, comments_file, output_file):
        """
        处理评论文件，提取VADER情感分数并保存

        Args:
            comments_file: 评论JSON文件路径
            output_file: 输出文件路径
        """
        print(f"处理评论文件: {comments_file}")

        # 加载评论数据
        with open(comments_file, 'r', encoding='utf-8') as f:
            comments = json.load(f)

        # 提取评论ID和文本
        comment_ids = [comment['commentId'] for comment in comments]
        comment_texts = [comment['text'] for comment in comments]

        print(f"共加载 {len(comment_texts)} 条评论")

        # 分析情感
        sentiment_scores = self.analyze_texts(comment_texts)

        # 创建结果字典
        result = {
            'comment_ids': comment_ids,
            'sentiment_scores': sentiment_scores
        }

        # 保存结果
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2)

        print(f"评论VADER情感分数已保存至: {output_file}")
        print(f"处理了 {len(sentiment_scores)} 条评论")

        # 打印一些统计信息
        compound_scores = [score['compound'] for score in sentiment_scores]
        print(f"情感分数统计:")
        print(f"  平均compound分数: {np.mean(compound_scores):.4f}")
        print(f"  compound分数标准差: {np.std(compound_scores):.4f}")
        print(f"  正面评论数量 (compound > 0.05): {sum(1 for s in compound_scores if s > 0.05)}")
        print(f"  中性评论数量 (-0.05 <= compound <= 0.05): {sum(1 for s in compound_scores if -0.05 <= s <= 0.05)}")
        print(f"  负面评论数量 (compound < -0.05): {sum(1 for s in compound_scores if s < -0.05)}")

    def process_media_sessions(self, media_file, output_file):
        """
        处理媒体会话文件，提取描述的VADER情感分数并保存

        Args:
            media_file: 媒体会话JSON文件路径
            output_file: 输出文件路径
        """
        print(f"处理媒体会话文件: {media_file}")

        # 加载媒体会话数据
        with open(media_file, 'r', encoding='utf-8') as f:
            media_sessions = json.load(f)

        # 提取媒体会话ID和描述
        media_ids = [session['postId'] for session in media_sessions]
        media_descriptions = [session.get('description', '') for session in media_sessions]

        print(f"共加载 {len(media_descriptions)} 个媒体会话")

        # 分析情感
        sentiment_scores = self.analyze_texts(media_descriptions)

        # 创建结果字典
        result = {
            'media_ids': media_ids,
            'sentiment_scores': sentiment_scores
        }

        # 保存结果
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2)

        print(f"媒体会话VADER情感分数已保存至: {output_file}")
        print(f"处理了 {len(sentiment_scores)} 个媒体会话")

        # 打印一些统计信息
        compound_scores = [score['compound'] for score in sentiment_scores]
        print(f"情感分数统计:")
        print(f"  平均compound分数: {np.mean(compound_scores):.4f}")
        print(f"  compound分数标准差: {np.std(compound_scores):.4f}")
        print(f"  正面媒体会话数量 (compound > 0.05): {sum(1 for s in compound_scores if s > 0.05)}")
        print(f"  中性媒体会话数量 (-0.05 <= compound <= 0.05): {sum(1 for s in compound_scores if -0.05 <= s <= 0.05)}")
        print(f"  负面媒体会话数量 (compound < -0.05): {sum(1 for s in compound_scores if s < -0.05)}")


def main():
    """主函数，用于处理评论和媒体会话数据"""
    print("开始VADER情感分析...")

    # 创建输出目录
    output_dir = "data/processed/vader_sentiment_scores"
    os.makedirs(output_dir, exist_ok=True)

    # 初始化VADER情感分析器
    analyzer = VaderSentimentAnalyzer()

    # 处理评论数据
    comments_file = "data/processed/integration/comments.json"
    comments_output = os.path.join(output_dir, "comment_sentiment_scores.json")

    if not os.path.exists(comments_output):
        print("开始处理评论数据...")
        analyzer.process_comments(comments_file, comments_output)
    else:
        print(f"评论情感分数文件已存在: {comments_output}")

    # 处理媒体会话数据
    media_file = "data/processed/integration/media_sessions.json"
    media_output = os.path.join(output_dir, "media_sentiment_scores.json")

    if not os.path.exists(media_output):
        print("开始处理媒体会话数据...")
        analyzer.process_media_sessions(media_file, media_output)
    else:
        print(f"媒体会话情感分数文件已存在: {media_output}")

    print("VADER情感分析完成!")


if __name__ == "__main__":
    main()